
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csfx2uw2tpg6g7s57a4zsv7zmmglucw3yw4mugjg6mhbahqcivg6.py
# Source Nodes: [word_emb_k], Original ATen: [aten.embedding]
# word_emb_k => embedding
triton_poi_fused_embedding_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 32000
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 32000), "index out of bounds: 0 <= tmp3 < 32000")
    tmp4 = tl.load(in_ptr1 + (x0 + (1024*tmp3)), None)
    tl.store(out_ptr0 + (x2), tmp4, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


cpp_fused_cat_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = c10::convert<long>(x0);
                        auto tmp7 = c10::convert<double>(tmp6);
                        auto tmp8 = static_cast<double>(-1.0);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = static_cast<double>(512.0);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = c10::convert<float>(tmp11);
                        auto tmp13 = c10::convert<long>(x1);
                        auto tmp14 = c10::convert<double>(tmp13);
                        auto tmp15 = static_cast<double>(2.0);
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp17 = static_cast<double>(0.0);
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp19 = c10::convert<float>(tmp18);
                        auto tmp20 = static_cast<float>(1024.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(10000.0);
                        auto tmp23 = std::pow(tmp22, tmp21);
                        auto tmp24 = 1 / tmp23;
                        auto tmp25 = static_cast<float>(1.0);
                        auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                        auto tmp27 = decltype(tmp12)(tmp12 * tmp26);
                        auto tmp28 = std::sin(tmp27);
                        return tmp28;
                    }
                    ;
                    auto tmp29 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp30 = tmp0 >= tmp3;
                    auto tmp31 = static_cast<long>(1024);
                    auto tmp32 = tmp0 < tmp31;
                    auto tmp33 = [&]
                    {
                        auto tmp34 = c10::convert<long>(x0);
                        auto tmp35 = c10::convert<double>(tmp34);
                        auto tmp36 = static_cast<double>(-1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = static_cast<double>(512.0);
                        auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                        auto tmp40 = c10::convert<float>(tmp39);
                        auto tmp41 = c10::convert<long>((-512L) + x1);
                        auto tmp42 = c10::convert<double>(tmp41);
                        auto tmp43 = static_cast<double>(2.0);
                        auto tmp44 = decltype(tmp42)(tmp42 * tmp43);
                        auto tmp45 = static_cast<double>(0.0);
                        auto tmp46 = decltype(tmp44)(tmp44 + tmp45);
                        auto tmp47 = c10::convert<float>(tmp46);
                        auto tmp48 = static_cast<float>(1024.0);
                        auto tmp49 = tmp47 / tmp48;
                        auto tmp50 = static_cast<float>(10000.0);
                        auto tmp51 = std::pow(tmp50, tmp49);
                        auto tmp52 = 1 / tmp51;
                        auto tmp53 = static_cast<float>(1.0);
                        auto tmp54 = decltype(tmp52)(tmp52 * tmp53);
                        auto tmp55 = decltype(tmp40)(tmp40 * tmp54);
                        auto tmp56 = std::cos(tmp55);
                        return tmp56;
                    }
                    ;
                    auto tmp57 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                    auto tmp58 = tmp4 ? tmp29 : tmp57;
                    out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp58;
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdpsnqsbk5755ogsinal4fuoqvlvbfc6qqxdubl3j4gvmr6jl3m.py
# Source Nodes: [add, add_1], Original ATen: [aten.add]
# add => add_2
# add_1 => add_3
triton_poi_fused_add_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktesaxyaovrdftydq7oc7akysamu74yybm4rvxfb5cj7shkgm5r.py
# Source Nodes: [arange_2], Original ATen: [aten.arange]
# arange_2 => iota_2
triton_poi_fused_arange_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cix6i3kwnlysecdyntqzcn67x2nbymylay543kwfh7ghhw2ckiwl.py
# Source Nodes: [add_2, add_3, attn_prob, attn_score, bd_1], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
# add_2 => add_4
# add_3 => add_5
# attn_prob => amax, div_1, exp, sub, sum_1
# attn_score => mul_4
# bd_1 => index
triton_red_fused__softmax_add_index_select_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.125
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = 0.0
        tmp14 = tmp12 + tmp13
        tmp15 = 0.125
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr1 + (512 + r2 + (1023*x0) + (524288*x1) + (524288*((r2 + (1023*x0)) // 523776))), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = 0.0
        tmp26 = tmp24 + tmp25
        tmp27 = 0.125
        tmp28 = tmp26 * tmp27
        tmp29 = tmp28 - tmp8
        tmp30 = tl.exp(tmp29)
        tmp31 = tmp30 / tmp20
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/csc53pzkz5obh663torqqeq7h7q3ge2yr2i4lmmjs6zp7qw4vjmi.py
# Source Nodes: [attn_out], Original ATen: [aten.clone]
# attn_out => clone
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32768*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (16*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckok66vpipnvffxbbqkwuden3r554yhr6qtlub2ma3asvy6yesng.py
# Source Nodes: [attn_out], Original ATen: [aten.clone]
# attn_out => clone_1
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 1024
    x2 = (xindex // 1024)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (1024*x1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16384*y0)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qxg733p6vcrmecc3xzykfwhzscbckvzhowcfyogblrzs7r2pat.py
# Source Nodes: [attn_out_2, output_1, output_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# attn_out_2 => add_6
# output_1 => add_7, add_8, mul_5, mul_6, rsqrt, sub_1, var_mean
# output_2 => view_34
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 1024.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwygaq6csoyawrxkvpdf2wixtdfsv4crw4uiuuyavhyaosfi4cho.py
# Source Nodes: [output_3], Original ATen: [aten.gelu]
# output_3 => add_9, erf, mul_7, mul_8, mul_9
triton_poi_fused_gelu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3xxn4ru7kxbhdwsxnj3mgii5yku6udokcm6dw7enug6yil7efz.py
# Source Nodes: [add_5, cat_2, output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_5 => add_10
# cat_2 => add_11, add_12, mul_10, mul_11, rsqrt_1, sub_2, var_mean_1
# output_1 => add_8, mul_6
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 1024, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 1024.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfiowx777zuno7ffv55evuu6j3lgdnmdjdxdmpoe7ct7q6afw36.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24, exp_24, log, sub_72, sub_73, sum_25
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (32000*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lpjngrwhe7ehcyiunqj7je6e54bao6vydiyoxezhsppd6b2aka.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_5, div_25, full_default_1, ne, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 32000
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 32000), "index out of bounds: 0 <= tmp12 < 32000")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp7.to(tl.float32)
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365 = args
    args.clear()
    assert_size_stride(primals_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_2, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_3, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_4, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_5, (16, 64), (64, 1))
    assert_size_stride(primals_6, (16, 64), (64, 1))
    assert_size_stride(primals_7, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_8, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_9, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_10, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_11, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_12, (16, 64), (64, 1))
    assert_size_stride(primals_13, (16, 64), (64, 1))
    assert_size_stride(primals_14, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_15, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_16, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_17, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_18, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_19, (16, 64), (64, 1))
    assert_size_stride(primals_20, (16, 64), (64, 1))
    assert_size_stride(primals_21, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_22, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_23, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_24, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_25, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_26, (16, 64), (64, 1))
    assert_size_stride(primals_27, (16, 64), (64, 1))
    assert_size_stride(primals_28, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_29, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_30, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_31, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_32, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_33, (16, 64), (64, 1))
    assert_size_stride(primals_34, (16, 64), (64, 1))
    assert_size_stride(primals_35, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_36, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_37, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_38, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_39, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_40, (16, 64), (64, 1))
    assert_size_stride(primals_41, (16, 64), (64, 1))
    assert_size_stride(primals_42, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_43, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_44, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_45, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_46, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_47, (16, 64), (64, 1))
    assert_size_stride(primals_48, (16, 64), (64, 1))
    assert_size_stride(primals_49, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_50, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_51, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_52, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_53, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_54, (16, 64), (64, 1))
    assert_size_stride(primals_55, (16, 64), (64, 1))
    assert_size_stride(primals_56, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_57, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_58, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_59, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_60, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_61, (16, 64), (64, 1))
    assert_size_stride(primals_62, (16, 64), (64, 1))
    assert_size_stride(primals_63, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_64, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_65, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_66, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_67, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_68, (16, 64), (64, 1))
    assert_size_stride(primals_69, (16, 64), (64, 1))
    assert_size_stride(primals_70, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_71, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_72, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_73, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_74, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_75, (16, 64), (64, 1))
    assert_size_stride(primals_76, (16, 64), (64, 1))
    assert_size_stride(primals_77, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_78, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_79, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_80, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_81, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_82, (16, 64), (64, 1))
    assert_size_stride(primals_83, (16, 64), (64, 1))
    assert_size_stride(primals_84, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_85, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_86, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_87, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_88, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_89, (16, 64), (64, 1))
    assert_size_stride(primals_90, (16, 64), (64, 1))
    assert_size_stride(primals_91, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_92, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_93, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_94, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_95, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_96, (16, 64), (64, 1))
    assert_size_stride(primals_97, (16, 64), (64, 1))
    assert_size_stride(primals_98, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_99, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_100, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_101, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_102, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_103, (16, 64), (64, 1))
    assert_size_stride(primals_104, (16, 64), (64, 1))
    assert_size_stride(primals_105, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_106, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_107, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_108, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_109, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_110, (16, 64), (64, 1))
    assert_size_stride(primals_111, (16, 64), (64, 1))
    assert_size_stride(primals_112, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_113, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_114, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_115, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_116, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_117, (16, 64), (64, 1))
    assert_size_stride(primals_118, (16, 64), (64, 1))
    assert_size_stride(primals_119, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_120, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_121, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_122, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_123, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_124, (16, 64), (64, 1))
    assert_size_stride(primals_125, (16, 64), (64, 1))
    assert_size_stride(primals_126, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_127, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_128, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_129, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_130, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_131, (16, 64), (64, 1))
    assert_size_stride(primals_132, (16, 64), (64, 1))
    assert_size_stride(primals_133, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_134, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_135, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_136, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_137, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_138, (16, 64), (64, 1))
    assert_size_stride(primals_139, (16, 64), (64, 1))
    assert_size_stride(primals_140, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_141, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_142, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_143, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_144, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_145, (16, 64), (64, 1))
    assert_size_stride(primals_146, (16, 64), (64, 1))
    assert_size_stride(primals_147, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_148, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_149, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_150, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_151, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_152, (16, 64), (64, 1))
    assert_size_stride(primals_153, (16, 64), (64, 1))
    assert_size_stride(primals_154, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_155, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_156, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_157, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_158, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_159, (16, 64), (64, 1))
    assert_size_stride(primals_160, (16, 64), (64, 1))
    assert_size_stride(primals_161, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_162, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_163, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_164, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_165, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_166, (16, 64), (64, 1))
    assert_size_stride(primals_167, (16, 64), (64, 1))
    assert_size_stride(primals_168, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(primals_169, (32000, 1024), (1024, 1))
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (4096, 1024), (1024, 1))
    assert_size_stride(primals_173, (4096, ), (1, ))
    assert_size_stride(primals_174, (1024, 4096), (4096, 1))
    assert_size_stride(primals_175, (1024, ), (1, ))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_179, (1024, ), (1, ))
    assert_size_stride(primals_180, (4096, 1024), (1024, 1))
    assert_size_stride(primals_181, (4096, ), (1, ))
    assert_size_stride(primals_182, (1024, 4096), (4096, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (4096, 1024), (1024, 1))
    assert_size_stride(primals_189, (4096, ), (1, ))
    assert_size_stride(primals_190, (1024, 4096), (4096, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_193, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (4096, 1024), (1024, 1))
    assert_size_stride(primals_197, (4096, ), (1, ))
    assert_size_stride(primals_198, (1024, 4096), (4096, 1))
    assert_size_stride(primals_199, (1024, ), (1, ))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, ), (1, ))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_203, (1024, ), (1, ))
    assert_size_stride(primals_204, (4096, 1024), (1024, 1))
    assert_size_stride(primals_205, (4096, ), (1, ))
    assert_size_stride(primals_206, (1024, 4096), (4096, 1))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, ), (1, ))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (1024, ), (1, ))
    assert_size_stride(primals_212, (4096, 1024), (1024, 1))
    assert_size_stride(primals_213, (4096, ), (1, ))
    assert_size_stride(primals_214, (1024, 4096), (4096, 1))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (4096, 1024), (1024, 1))
    assert_size_stride(primals_221, (4096, ), (1, ))
    assert_size_stride(primals_222, (1024, 4096), (4096, 1))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_225, (1024, ), (1, ))
    assert_size_stride(primals_226, (1024, ), (1, ))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (4096, 1024), (1024, 1))
    assert_size_stride(primals_229, (4096, ), (1, ))
    assert_size_stride(primals_230, (1024, 4096), (4096, 1))
    assert_size_stride(primals_231, (1024, ), (1, ))
    assert_size_stride(primals_232, (1024, ), (1, ))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, ), (1, ))
    assert_size_stride(primals_235, (1024, ), (1, ))
    assert_size_stride(primals_236, (4096, 1024), (1024, 1))
    assert_size_stride(primals_237, (4096, ), (1, ))
    assert_size_stride(primals_238, (1024, 4096), (4096, 1))
    assert_size_stride(primals_239, (1024, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (4096, 1024), (1024, 1))
    assert_size_stride(primals_245, (4096, ), (1, ))
    assert_size_stride(primals_246, (1024, 4096), (4096, 1))
    assert_size_stride(primals_247, (1024, ), (1, ))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_252, (4096, 1024), (1024, 1))
    assert_size_stride(primals_253, (4096, ), (1, ))
    assert_size_stride(primals_254, (1024, 4096), (4096, 1))
    assert_size_stride(primals_255, (1024, ), (1, ))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_257, (1024, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (4096, 1024), (1024, 1))
    assert_size_stride(primals_261, (4096, ), (1, ))
    assert_size_stride(primals_262, (1024, 4096), (4096, 1))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_265, (1024, ), (1, ))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_267, (1024, ), (1, ))
    assert_size_stride(primals_268, (4096, 1024), (1024, 1))
    assert_size_stride(primals_269, (4096, ), (1, ))
    assert_size_stride(primals_270, (1024, 4096), (4096, 1))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_273, (1024, ), (1, ))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (4096, 1024), (1024, 1))
    assert_size_stride(primals_277, (4096, ), (1, ))
    assert_size_stride(primals_278, (1024, 4096), (4096, 1))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (1024, ), (1, ))
    assert_size_stride(primals_284, (4096, 1024), (1024, 1))
    assert_size_stride(primals_285, (4096, ), (1, ))
    assert_size_stride(primals_286, (1024, 4096), (4096, 1))
    assert_size_stride(primals_287, (1024, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (4096, 1024), (1024, 1))
    assert_size_stride(primals_293, (4096, ), (1, ))
    assert_size_stride(primals_294, (1024, 4096), (4096, 1))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (1024, ), (1, ))
    assert_size_stride(primals_298, (1024, ), (1, ))
    assert_size_stride(primals_299, (1024, ), (1, ))
    assert_size_stride(primals_300, (4096, 1024), (1024, 1))
    assert_size_stride(primals_301, (4096, ), (1, ))
    assert_size_stride(primals_302, (1024, 4096), (4096, 1))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_308, (4096, 1024), (1024, 1))
    assert_size_stride(primals_309, (4096, ), (1, ))
    assert_size_stride(primals_310, (1024, 4096), (4096, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (4096, 1024), (1024, 1))
    assert_size_stride(primals_317, (4096, ), (1, ))
    assert_size_stride(primals_318, (1024, 4096), (4096, 1))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (1024, ), (1, ))
    assert_size_stride(primals_323, (1024, ), (1, ))
    assert_size_stride(primals_324, (4096, 1024), (1024, 1))
    assert_size_stride(primals_325, (4096, ), (1, ))
    assert_size_stride(primals_326, (1024, 4096), (4096, 1))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, ), (1, ))
    assert_size_stride(primals_329, (1024, ), (1, ))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, ), (1, ))
    assert_size_stride(primals_332, (4096, 1024), (1024, 1))
    assert_size_stride(primals_333, (4096, ), (1, ))
    assert_size_stride(primals_334, (1024, 4096), (4096, 1))
    assert_size_stride(primals_335, (1024, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (1024, ), (1, ))
    assert_size_stride(primals_340, (4096, 1024), (1024, 1))
    assert_size_stride(primals_341, (4096, ), (1, ))
    assert_size_stride(primals_342, (1024, 4096), (4096, 1))
    assert_size_stride(primals_343, (1024, ), (1, ))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_345, (1024, ), (1, ))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (4096, 1024), (1024, 1))
    assert_size_stride(primals_349, (4096, ), (1, ))
    assert_size_stride(primals_350, (1024, 4096), (4096, 1))
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (1024, ), (1, ))
    assert_size_stride(primals_354, (1024, ), (1, ))
    assert_size_stride(primals_355, (1024, ), (1, ))
    assert_size_stride(primals_356, (4096, 1024), (1024, 1))
    assert_size_stride(primals_357, (4096, ), (1, ))
    assert_size_stride(primals_358, (1024, 4096), (4096, 1))
    assert_size_stride(primals_359, (1024, ), (1, ))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (32000, 1024), (1024, 1))
    assert_size_stride(primals_363, (32000, ), (1, ))
    assert_size_stride(primals_364, (1, 512), (512, 1))
    assert_size_stride(primals_365, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((512, 1, 1024), (1024, 524288, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [word_emb_k], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(primals_364, primals_169, buf0, 524288, grid=grid(524288), stream=stream0)
        del primals_169
        # Source Nodes: [cat_1, word_emb_k], Original ATen: [aten.embedding, aten.native_dropout]
        buf1 = aten.native_dropout(buf0, 0.1, True)
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
    buf4 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_cat_1(c_void_p(buf4.data_ptr()))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf5 = empty((1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf5.copy_(reinterpret_tensor(buf4, (1024, 1, 1024), (1024, 0, 1), 0))
        del buf4
        # Source Nodes: [pos_emb_6], Original ATen: [aten.native_dropout]
        buf6 = aten.native_dropout(buf5, 0.1, True)
        buf7 = buf6[0]
        del buf6
        buf9 = reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0); del buf0  # reuse
        # Source Nodes: [q_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf9)
        buf10 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_2, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf10)
        buf11 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(primals_3, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf11)
        buf12 = reinterpret_tensor(buf5, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf5  # reuse
        # Source Nodes: [k_head_r], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_4, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf12)
        del primals_4
        buf13 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf15 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf9, primals_5, primals_6, buf13, buf15, 524288, grid=grid(524288), stream=stream0)
        del primals_5
        del primals_6
        buf14 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [ac], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf10, (16, 64, 512), (64, 1, 1024), 0), out=buf14)
        buf16 = empty((16, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [bd], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf12, (16, 64, 1024), (64, 1, 1024), 0), out=buf16)
        buf17 = empty((512, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [arange_2], Original ATen: [aten.arange]
        triton_poi_fused_arange_3.run(buf17, 512, grid=grid(512), stream=stream0)
        buf20 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, attn_prob, attn_score, bd_1], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf14, buf16, buf20, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_2, add_3, attn_prob, attn_prob_1, attn_score, bd_1], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf21 = aten.native_dropout(buf20, 0.1, True)
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = reinterpret_tensor(buf9, (16, 512, 64), (32768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [attn_vec], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf11, (16, 512, 64), (64, 1024, 1), 0), out=buf24)
        buf25 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf24, buf25, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf26 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_7, buf26, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_7
        buf27 = reinterpret_tensor(buf24, (1, 512, 1024), (524288, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [attn_out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf26, (1, 1024, 1024), (0, 1024, 1), 0), out=buf27)
        # Source Nodes: [attn_out_1], Original ATen: [aten.native_dropout]
        buf28 = aten.native_dropout(reinterpret_tensor(buf27, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf34 = reinterpret_tensor(buf27, (512, 1, 1024), (1024, 1024, 1), 0); del buf27  # reuse
        buf35 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1027 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_2, output_1, output_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf29, buf2, primals_170, primals_171, buf34, buf35, buf1027, 512, 1024, grid=grid(512), stream=stream0)
        buf36 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf35, reinterpret_tensor(primals_172, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf36)
        del primals_173
        buf37 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf36, buf37, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_3, output_4], Original ATen: [aten.gelu, aten.native_dropout]
        buf38 = aten.native_dropout(buf37, 0.1, True)
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = reinterpret_tensor(buf29, (512, 1024), (1024, 1), 0); del buf29  # reuse
        # Source Nodes: [output_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_175, reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_174, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf41)
        del primals_175
        # Source Nodes: [output_6], Original ATen: [aten.native_dropout]
        buf42 = aten.native_dropout(reinterpret_tensor(buf41, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf48 = reinterpret_tensor(buf41, (512, 1, 1024), (1024, 1024, 1), 0); del buf41  # reuse
        buf49 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1026 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, cat_2, output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf43, buf34, primals_170, primals_171, primals_176, primals_177, buf48, buf49, buf1026, 512, 1024, grid=grid(512), stream=stream0)
        del primals_171
        del primals_177
        buf50 = reinterpret_tensor(buf43, (1, 512, 1024), (524288, 1024, 1), 0); del buf43  # reuse
        # Source Nodes: [q_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_8, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf50)
        buf51 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_9, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf51)
        buf52 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_10, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf52)
        buf53 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_11, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf53)
        del primals_11
        buf54 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf56 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf50, primals_12, primals_13, buf54, buf56, 524288, grid=grid(524288), stream=stream0)
        del primals_12
        del primals_13
        buf55 = buf14; del buf14  # reuse
        # Source Nodes: [ac_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf51, (16, 64, 512), (64, 1, 1024), 0), out=buf55)
        buf57 = buf16; del buf16  # reuse
        # Source Nodes: [bd_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf53, (16, 64, 1024), (64, 1, 1024), 0), out=buf57)
        buf60 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, add_9, attn_prob_2, attn_score_1, bd_3], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf55, buf57, buf60, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_8, add_9, attn_prob_2, attn_prob_3, attn_score_1, bd_3], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf61 = aten.native_dropout(buf60, 0.1, True)
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf64 = reinterpret_tensor(buf50, (16, 512, 64), (32768, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [attn_vec_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf52, (16, 512, 64), (64, 1024, 1), 0), out=buf64)
        buf65 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf64, buf65, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf66 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_14, buf66, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_14
        buf67 = reinterpret_tensor(buf64, (1, 512, 1024), (524288, 1024, 1), 0); del buf64  # reuse
        # Source Nodes: [attn_out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf66, (1, 1024, 1024), (0, 1024, 1), 0), out=buf67)
        # Source Nodes: [attn_out_4], Original ATen: [aten.native_dropout]
        buf68 = aten.native_dropout(reinterpret_tensor(buf67, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf69 = buf68[0]
        buf70 = buf68[1]
        del buf68
        buf74 = reinterpret_tensor(buf67, (512, 1, 1024), (1024, 1024, 1), 0); del buf67  # reuse
        buf75 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1025 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_5, output_10, output_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf69, buf49, primals_178, primals_179, buf74, buf75, buf1025, 512, 1024, grid=grid(512), stream=stream0)
        buf76 = reinterpret_tensor(buf37, (512, 4096), (4096, 1), 0); del buf37  # reuse
        # Source Nodes: [output_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, buf75, reinterpret_tensor(primals_180, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf76)
        del primals_181
        buf77 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf76, buf77, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_11, output_12], Original ATen: [aten.gelu, aten.native_dropout]
        buf78 = aten.native_dropout(buf77, 0.1, True)
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [output_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_183, reinterpret_tensor(buf79, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_182, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf81)
        del primals_183
        # Source Nodes: [output_14], Original ATen: [aten.native_dropout]
        buf82 = aten.native_dropout(reinterpret_tensor(buf81, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf83 = buf82[0]
        buf84 = buf82[1]
        del buf82
        buf88 = reinterpret_tensor(buf81, (512, 1, 1024), (1024, 1024, 1), 0); del buf81  # reuse
        buf89 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1024 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, cat_3, output_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf83, buf74, primals_178, primals_179, primals_184, primals_185, buf88, buf89, buf1024, 512, 1024, grid=grid(512), stream=stream0)
        del primals_179
        del primals_185
        buf90 = reinterpret_tensor(buf83, (1, 512, 1024), (524288, 1024, 1), 0); del buf83  # reuse
        # Source Nodes: [q_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_15, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf90)
        buf91 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_16, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf91)
        buf92 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_17, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf92)
        buf93 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_18, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf93)
        del primals_18
        buf94 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf96 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, add_13], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf90, primals_19, primals_20, buf94, buf96, 524288, grid=grid(524288), stream=stream0)
        del primals_19
        del primals_20
        buf95 = buf55; del buf55  # reuse
        # Source Nodes: [ac_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf91, (16, 64, 512), (64, 1, 1024), 0), out=buf95)
        buf97 = buf57; del buf57  # reuse
        # Source Nodes: [bd_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf93, (16, 64, 1024), (64, 1, 1024), 0), out=buf97)
        buf100 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, add_15, attn_prob_4, attn_score_2, bd_5], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf95, buf97, buf100, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_14, add_15, attn_prob_4, attn_prob_5, attn_score_2, bd_5], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf101 = aten.native_dropout(buf100, 0.1, True)
        buf102 = buf101[0]
        buf103 = buf101[1]
        del buf101
        buf104 = reinterpret_tensor(buf90, (16, 512, 64), (32768, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [attn_vec_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf92, (16, 512, 64), (64, 1024, 1), 0), out=buf104)
        buf105 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf104, buf105, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf106 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_21, buf106, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_21
        buf107 = reinterpret_tensor(buf104, (1, 512, 1024), (524288, 1024, 1), 0); del buf104  # reuse
        # Source Nodes: [attn_out_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf106, (1, 1024, 1024), (0, 1024, 1), 0), out=buf107)
        # Source Nodes: [attn_out_7], Original ATen: [aten.native_dropout]
        buf108 = aten.native_dropout(reinterpret_tensor(buf107, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf109 = buf108[0]
        buf110 = buf108[1]
        del buf108
        buf114 = reinterpret_tensor(buf107, (512, 1, 1024), (1024, 1024, 1), 0); del buf107  # reuse
        buf115 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1023 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_8, output_17, output_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf109, buf89, primals_186, primals_187, buf114, buf115, buf1023, 512, 1024, grid=grid(512), stream=stream0)
        buf116 = reinterpret_tensor(buf77, (512, 4096), (4096, 1), 0); del buf77  # reuse
        # Source Nodes: [output_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf115, reinterpret_tensor(primals_188, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf116)
        del primals_189
        buf117 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_19], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf116, buf117, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_19, output_20], Original ATen: [aten.gelu, aten.native_dropout]
        buf118 = aten.native_dropout(buf117, 0.1, True)
        buf119 = buf118[0]
        buf120 = buf118[1]
        del buf118
        buf121 = reinterpret_tensor(buf109, (512, 1024), (1024, 1), 0); del buf109  # reuse
        # Source Nodes: [output_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_191, reinterpret_tensor(buf119, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_190, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf121)
        del primals_191
        # Source Nodes: [output_22], Original ATen: [aten.native_dropout]
        buf122 = aten.native_dropout(reinterpret_tensor(buf121, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf128 = reinterpret_tensor(buf121, (512, 1, 1024), (1024, 1024, 1), 0); del buf121  # reuse
        buf129 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1022 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, cat_4, output_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf123, buf114, primals_186, primals_187, primals_192, primals_193, buf128, buf129, buf1022, 512, 1024, grid=grid(512), stream=stream0)
        del primals_187
        del primals_193
        buf130 = reinterpret_tensor(buf123, (1, 512, 1024), (524288, 1024, 1), 0); del buf123  # reuse
        # Source Nodes: [q_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_22, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf130)
        buf131 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_23, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf131)
        buf132 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_24, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf132)
        buf133 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_25, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf133)
        del primals_25
        buf134 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf136 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf130, primals_26, primals_27, buf134, buf136, 524288, grid=grid(524288), stream=stream0)
        del primals_26
        del primals_27
        buf135 = buf95; del buf95  # reuse
        # Source Nodes: [ac_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf131, (16, 64, 512), (64, 1, 1024), 0), out=buf135)
        buf137 = buf97; del buf97  # reuse
        # Source Nodes: [bd_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf133, (16, 64, 1024), (64, 1, 1024), 0), out=buf137)
        buf140 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, add_21, attn_prob_6, attn_score_3, bd_7], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf135, buf137, buf140, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_20, add_21, attn_prob_6, attn_prob_7, attn_score_3, bd_7], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf141 = aten.native_dropout(buf140, 0.1, True)
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = reinterpret_tensor(buf130, (16, 512, 64), (32768, 64, 1), 0); del buf130  # reuse
        # Source Nodes: [attn_vec_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf132, (16, 512, 64), (64, 1024, 1), 0), out=buf144)
        buf145 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf144, buf145, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf146 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_28, buf146, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_28
        buf147 = reinterpret_tensor(buf144, (1, 512, 1024), (524288, 1024, 1), 0); del buf144  # reuse
        # Source Nodes: [attn_out_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf146, (1, 1024, 1024), (0, 1024, 1), 0), out=buf147)
        # Source Nodes: [attn_out_10], Original ATen: [aten.native_dropout]
        buf148 = aten.native_dropout(reinterpret_tensor(buf147, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf149 = buf148[0]
        buf150 = buf148[1]
        del buf148
        buf154 = reinterpret_tensor(buf147, (512, 1, 1024), (1024, 1024, 1), 0); del buf147  # reuse
        buf155 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1021 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_11, output_25, output_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf149, buf129, primals_194, primals_195, buf154, buf155, buf1021, 512, 1024, grid=grid(512), stream=stream0)
        buf156 = reinterpret_tensor(buf117, (512, 4096), (4096, 1), 0); del buf117  # reuse
        # Source Nodes: [output_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf155, reinterpret_tensor(primals_196, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf156)
        del primals_197
        buf157 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf156, buf157, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_27, output_28], Original ATen: [aten.gelu, aten.native_dropout]
        buf158 = aten.native_dropout(buf157, 0.1, True)
        buf159 = buf158[0]
        buf160 = buf158[1]
        del buf158
        buf161 = reinterpret_tensor(buf149, (512, 1024), (1024, 1), 0); del buf149  # reuse
        # Source Nodes: [output_29], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, reinterpret_tensor(buf159, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_198, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf161)
        del primals_199
        # Source Nodes: [output_30], Original ATen: [aten.native_dropout]
        buf162 = aten.native_dropout(reinterpret_tensor(buf161, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf168 = reinterpret_tensor(buf161, (512, 1, 1024), (1024, 1024, 1), 0); del buf161  # reuse
        buf169 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1020 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, cat_5, output_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf163, buf154, primals_194, primals_195, primals_200, primals_201, buf168, buf169, buf1020, 512, 1024, grid=grid(512), stream=stream0)
        del primals_195
        del primals_201
        buf170 = reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0); del buf163  # reuse
        # Source Nodes: [q_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_29, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf170)
        buf171 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_30, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf171)
        buf172 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_31, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf172)
        buf173 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_32, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf173)
        del primals_32
        buf174 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf176 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, add_25], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf170, primals_33, primals_34, buf174, buf176, 524288, grid=grid(524288), stream=stream0)
        del primals_33
        del primals_34
        buf175 = buf135; del buf135  # reuse
        # Source Nodes: [ac_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf171, (16, 64, 512), (64, 1, 1024), 0), out=buf175)
        buf177 = buf137; del buf137  # reuse
        # Source Nodes: [bd_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf173, (16, 64, 1024), (64, 1, 1024), 0), out=buf177)
        buf180 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, add_27, attn_prob_8, attn_score_4, bd_9], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf175, buf177, buf180, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_26, add_27, attn_prob_8, attn_prob_9, attn_score_4, bd_9], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf181 = aten.native_dropout(buf180, 0.1, True)
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = reinterpret_tensor(buf170, (16, 512, 64), (32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [attn_vec_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf172, (16, 512, 64), (64, 1024, 1), 0), out=buf184)
        buf185 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf184, buf185, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf186 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_35, buf186, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_35
        buf187 = reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0); del buf184  # reuse
        # Source Nodes: [attn_out_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf185, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf186, (1, 1024, 1024), (0, 1024, 1), 0), out=buf187)
        # Source Nodes: [attn_out_13], Original ATen: [aten.native_dropout]
        buf188 = aten.native_dropout(reinterpret_tensor(buf187, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf194 = reinterpret_tensor(buf187, (512, 1, 1024), (1024, 1024, 1), 0); del buf187  # reuse
        buf195 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1019 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_14, output_33, output_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf189, buf169, primals_202, primals_203, buf194, buf195, buf1019, 512, 1024, grid=grid(512), stream=stream0)
        buf196 = reinterpret_tensor(buf157, (512, 4096), (4096, 1), 0); del buf157  # reuse
        # Source Nodes: [output_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, buf195, reinterpret_tensor(primals_204, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf196)
        del primals_205
        buf197 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf196, buf197, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_35, output_36], Original ATen: [aten.gelu, aten.native_dropout]
        buf198 = aten.native_dropout(buf197, 0.1, True)
        buf199 = buf198[0]
        buf200 = buf198[1]
        del buf198
        buf201 = reinterpret_tensor(buf189, (512, 1024), (1024, 1), 0); del buf189  # reuse
        # Source Nodes: [output_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_207, reinterpret_tensor(buf199, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_206, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf201)
        del primals_207
        # Source Nodes: [output_38], Original ATen: [aten.native_dropout]
        buf202 = aten.native_dropout(reinterpret_tensor(buf201, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf203 = buf202[0]
        buf204 = buf202[1]
        del buf202
        buf208 = reinterpret_tensor(buf201, (512, 1, 1024), (1024, 1024, 1), 0); del buf201  # reuse
        buf209 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1018 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, cat_6, output_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf203, buf194, primals_202, primals_203, primals_208, primals_209, buf208, buf209, buf1018, 512, 1024, grid=grid(512), stream=stream0)
        del primals_203
        del primals_209
        buf210 = reinterpret_tensor(buf203, (1, 512, 1024), (524288, 1024, 1), 0); del buf203  # reuse
        # Source Nodes: [q_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_36, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf210)
        buf211 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_37, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf211)
        buf212 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_38, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf212)
        buf213 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_39, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf213)
        del primals_39
        buf214 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf216 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf210, primals_40, primals_41, buf214, buf216, 524288, grid=grid(524288), stream=stream0)
        del primals_40
        del primals_41
        buf215 = buf175; del buf175  # reuse
        # Source Nodes: [ac_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf211, (16, 64, 512), (64, 1, 1024), 0), out=buf215)
        buf217 = buf177; del buf177  # reuse
        # Source Nodes: [bd_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf213, (16, 64, 1024), (64, 1, 1024), 0), out=buf217)
        buf220 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, add_33, attn_prob_10, attn_score_5, bd_11], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf215, buf217, buf220, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_32, add_33, attn_prob_10, attn_prob_11, attn_score_5, bd_11], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf221 = aten.native_dropout(buf220, 0.1, True)
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = reinterpret_tensor(buf210, (16, 512, 64), (32768, 64, 1), 0); del buf210  # reuse
        # Source Nodes: [attn_vec_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf222, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf212, (16, 512, 64), (64, 1024, 1), 0), out=buf224)
        buf225 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf224, buf225, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf226 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_42, buf226, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_42
        buf227 = reinterpret_tensor(buf224, (1, 512, 1024), (524288, 1024, 1), 0); del buf224  # reuse
        # Source Nodes: [attn_out_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf226, (1, 1024, 1024), (0, 1024, 1), 0), out=buf227)
        # Source Nodes: [attn_out_16], Original ATen: [aten.native_dropout]
        buf228 = aten.native_dropout(reinterpret_tensor(buf227, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf229 = buf228[0]
        buf230 = buf228[1]
        del buf228
        buf234 = reinterpret_tensor(buf227, (512, 1, 1024), (1024, 1024, 1), 0); del buf227  # reuse
        buf235 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1017 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_17, output_41, output_42], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf229, buf209, primals_210, primals_211, buf234, buf235, buf1017, 512, 1024, grid=grid(512), stream=stream0)
        buf236 = reinterpret_tensor(buf197, (512, 4096), (4096, 1), 0); del buf197  # reuse
        # Source Nodes: [output_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, buf235, reinterpret_tensor(primals_212, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf236)
        del primals_213
        buf237 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_43], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf236, buf237, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_43, output_44], Original ATen: [aten.gelu, aten.native_dropout]
        buf238 = aten.native_dropout(buf237, 0.1, True)
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = reinterpret_tensor(buf229, (512, 1024), (1024, 1), 0); del buf229  # reuse
        # Source Nodes: [output_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_215, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_214, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf241)
        del primals_215
        # Source Nodes: [output_46], Original ATen: [aten.native_dropout]
        buf242 = aten.native_dropout(reinterpret_tensor(buf241, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        buf248 = reinterpret_tensor(buf241, (512, 1, 1024), (1024, 1024, 1), 0); del buf241  # reuse
        buf249 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1016 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, cat_7, output_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf243, buf234, primals_210, primals_211, primals_216, primals_217, buf248, buf249, buf1016, 512, 1024, grid=grid(512), stream=stream0)
        del primals_211
        del primals_217
        buf250 = reinterpret_tensor(buf243, (1, 512, 1024), (524288, 1024, 1), 0); del buf243  # reuse
        # Source Nodes: [q_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_43, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf250)
        buf251 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_44, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf251)
        buf252 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_45, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf252)
        buf253 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_46, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf253)
        del primals_46
        buf254 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf256 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, add_37], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf250, primals_47, primals_48, buf254, buf256, 524288, grid=grid(524288), stream=stream0)
        del primals_47
        del primals_48
        buf255 = buf215; del buf215  # reuse
        # Source Nodes: [ac_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf254, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf251, (16, 64, 512), (64, 1, 1024), 0), out=buf255)
        buf257 = buf217; del buf217  # reuse
        # Source Nodes: [bd_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf256, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf253, (16, 64, 1024), (64, 1, 1024), 0), out=buf257)
        buf260 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, attn_prob_12, attn_score_6, bd_13], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf255, buf257, buf260, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_38, add_39, attn_prob_12, attn_prob_13, attn_score_6, bd_13], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf261 = aten.native_dropout(buf260, 0.1, True)
        buf262 = buf261[0]
        buf263 = buf261[1]
        del buf261
        buf264 = reinterpret_tensor(buf250, (16, 512, 64), (32768, 64, 1), 0); del buf250  # reuse
        # Source Nodes: [attn_vec_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf252, (16, 512, 64), (64, 1024, 1), 0), out=buf264)
        buf265 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf264, buf265, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf266 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_49, buf266, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_49
        buf267 = reinterpret_tensor(buf264, (1, 512, 1024), (524288, 1024, 1), 0); del buf264  # reuse
        # Source Nodes: [attn_out_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf266, (1, 1024, 1024), (0, 1024, 1), 0), out=buf267)
        # Source Nodes: [attn_out_19], Original ATen: [aten.native_dropout]
        buf268 = aten.native_dropout(reinterpret_tensor(buf267, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf274 = reinterpret_tensor(buf267, (512, 1, 1024), (1024, 1024, 1), 0); del buf267  # reuse
        buf275 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1015 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_20, output_49, output_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf269, buf249, primals_218, primals_219, buf274, buf275, buf1015, 512, 1024, grid=grid(512), stream=stream0)
        buf276 = reinterpret_tensor(buf237, (512, 4096), (4096, 1), 0); del buf237  # reuse
        # Source Nodes: [output_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_221, buf275, reinterpret_tensor(primals_220, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf276)
        del primals_221
        buf277 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf276, buf277, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_51, output_52], Original ATen: [aten.gelu, aten.native_dropout]
        buf278 = aten.native_dropout(buf277, 0.1, True)
        buf279 = buf278[0]
        buf280 = buf278[1]
        del buf278
        buf281 = reinterpret_tensor(buf269, (512, 1024), (1024, 1), 0); del buf269  # reuse
        # Source Nodes: [output_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_223, reinterpret_tensor(buf279, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_222, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf281)
        del primals_223
        # Source Nodes: [output_54], Original ATen: [aten.native_dropout]
        buf282 = aten.native_dropout(reinterpret_tensor(buf281, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        buf288 = reinterpret_tensor(buf281, (512, 1, 1024), (1024, 1024, 1), 0); del buf281  # reuse
        buf289 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1014 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_41, cat_8, output_49], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf283, buf274, primals_218, primals_219, primals_224, primals_225, buf288, buf289, buf1014, 512, 1024, grid=grid(512), stream=stream0)
        del primals_219
        del primals_225
        buf290 = reinterpret_tensor(buf283, (1, 512, 1024), (524288, 1024, 1), 0); del buf283  # reuse
        # Source Nodes: [q_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_50, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf290)
        buf291 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_51, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf291)
        buf292 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_52, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf292)
        buf293 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_53, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf293)
        del primals_53
        buf294 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf296 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf290, primals_54, primals_55, buf294, buf296, 524288, grid=grid(524288), stream=stream0)
        del primals_54
        del primals_55
        buf295 = buf255; del buf255  # reuse
        # Source Nodes: [ac_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf291, (16, 64, 512), (64, 1, 1024), 0), out=buf295)
        buf297 = buf257; del buf257  # reuse
        # Source Nodes: [bd_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf293, (16, 64, 1024), (64, 1, 1024), 0), out=buf297)
        buf300 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, add_45, attn_prob_14, attn_score_7, bd_15], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf295, buf297, buf300, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_44, add_45, attn_prob_14, attn_prob_15, attn_score_7, bd_15], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf301 = aten.native_dropout(buf300, 0.1, True)
        buf302 = buf301[0]
        buf303 = buf301[1]
        del buf301
        buf304 = reinterpret_tensor(buf290, (16, 512, 64), (32768, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [attn_vec_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf292, (16, 512, 64), (64, 1024, 1), 0), out=buf304)
        buf305 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf304, buf305, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf306 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_56, buf306, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_56
        buf307 = reinterpret_tensor(buf304, (1, 512, 1024), (524288, 1024, 1), 0); del buf304  # reuse
        # Source Nodes: [attn_out_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf306, (1, 1024, 1024), (0, 1024, 1), 0), out=buf307)
        # Source Nodes: [attn_out_22], Original ATen: [aten.native_dropout]
        buf308 = aten.native_dropout(reinterpret_tensor(buf307, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf314 = reinterpret_tensor(buf307, (512, 1, 1024), (1024, 1024, 1), 0); del buf307  # reuse
        buf315 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1013 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_23, output_57, output_58], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf309, buf289, primals_226, primals_227, buf314, buf315, buf1013, 512, 1024, grid=grid(512), stream=stream0)
        buf316 = reinterpret_tensor(buf277, (512, 4096), (4096, 1), 0); del buf277  # reuse
        # Source Nodes: [output_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_229, buf315, reinterpret_tensor(primals_228, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf316)
        del primals_229
        buf317 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_59], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf316, buf317, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_59, output_60], Original ATen: [aten.gelu, aten.native_dropout]
        buf318 = aten.native_dropout(buf317, 0.1, True)
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf321 = reinterpret_tensor(buf309, (512, 1024), (1024, 1), 0); del buf309  # reuse
        # Source Nodes: [output_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_231, reinterpret_tensor(buf319, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_230, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf321)
        del primals_231
        # Source Nodes: [output_62], Original ATen: [aten.native_dropout]
        buf322 = aten.native_dropout(reinterpret_tensor(buf321, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf323 = buf322[0]
        buf324 = buf322[1]
        del buf322
        buf328 = reinterpret_tensor(buf321, (512, 1, 1024), (1024, 1024, 1), 0); del buf321  # reuse
        buf329 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1012 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47, cat_9, output_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf323, buf314, primals_226, primals_227, primals_232, primals_233, buf328, buf329, buf1012, 512, 1024, grid=grid(512), stream=stream0)
        del primals_227
        del primals_233
        buf330 = reinterpret_tensor(buf323, (1, 512, 1024), (524288, 1024, 1), 0); del buf323  # reuse
        # Source Nodes: [q_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_57, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf330)
        buf331 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_58, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf331)
        buf332 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_59, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf332)
        buf333 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_60, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf333)
        del primals_60
        buf334 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf336 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, add_49], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf330, primals_61, primals_62, buf334, buf336, 524288, grid=grid(524288), stream=stream0)
        del primals_61
        del primals_62
        buf335 = buf295; del buf295  # reuse
        # Source Nodes: [ac_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf334, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf331, (16, 64, 512), (64, 1, 1024), 0), out=buf335)
        buf337 = buf297; del buf297  # reuse
        # Source Nodes: [bd_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf336, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf333, (16, 64, 1024), (64, 1, 1024), 0), out=buf337)
        buf340 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, add_51, attn_prob_16, attn_score_8, bd_17], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf335, buf337, buf340, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_50, add_51, attn_prob_16, attn_prob_17, attn_score_8, bd_17], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf341 = aten.native_dropout(buf340, 0.1, True)
        buf342 = buf341[0]
        buf343 = buf341[1]
        del buf341
        buf344 = reinterpret_tensor(buf330, (16, 512, 64), (32768, 64, 1), 0); del buf330  # reuse
        # Source Nodes: [attn_vec_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf332, (16, 512, 64), (64, 1024, 1), 0), out=buf344)
        buf345 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf344, buf345, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf346 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_63, buf346, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_63
        buf347 = reinterpret_tensor(buf344, (1, 512, 1024), (524288, 1024, 1), 0); del buf344  # reuse
        # Source Nodes: [attn_out_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf345, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf346, (1, 1024, 1024), (0, 1024, 1), 0), out=buf347)
        # Source Nodes: [attn_out_25], Original ATen: [aten.native_dropout]
        buf348 = aten.native_dropout(reinterpret_tensor(buf347, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf354 = reinterpret_tensor(buf347, (512, 1, 1024), (1024, 1024, 1), 0); del buf347  # reuse
        buf355 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1011 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_26, output_65, output_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf349, buf329, primals_234, primals_235, buf354, buf355, buf1011, 512, 1024, grid=grid(512), stream=stream0)
        buf356 = reinterpret_tensor(buf317, (512, 4096), (4096, 1), 0); del buf317  # reuse
        # Source Nodes: [output_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_237, buf355, reinterpret_tensor(primals_236, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf356)
        del primals_237
        buf357 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_67], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf356, buf357, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_67, output_68], Original ATen: [aten.gelu, aten.native_dropout]
        buf358 = aten.native_dropout(buf357, 0.1, True)
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = reinterpret_tensor(buf349, (512, 1024), (1024, 1), 0); del buf349  # reuse
        # Source Nodes: [output_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_239, reinterpret_tensor(buf359, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_238, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf361)
        del primals_239
        # Source Nodes: [output_70], Original ATen: [aten.native_dropout]
        buf362 = aten.native_dropout(reinterpret_tensor(buf361, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf368 = reinterpret_tensor(buf361, (512, 1, 1024), (1024, 1024, 1), 0); del buf361  # reuse
        buf369 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1010 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, cat_10, output_65], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf363, buf354, primals_234, primals_235, primals_240, primals_241, buf368, buf369, buf1010, 512, 1024, grid=grid(512), stream=stream0)
        del primals_235
        del primals_241
        buf370 = reinterpret_tensor(buf363, (1, 512, 1024), (524288, 1024, 1), 0); del buf363  # reuse
        # Source Nodes: [q_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_64, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf370)
        buf371 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_65, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf371)
        buf372 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_66, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf372)
        buf373 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_67, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf373)
        del primals_67
        buf374 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf376 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_54, add_55], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf370, primals_68, primals_69, buf374, buf376, 524288, grid=grid(524288), stream=stream0)
        del primals_68
        del primals_69
        buf375 = buf335; del buf335  # reuse
        # Source Nodes: [ac_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf371, (16, 64, 512), (64, 1, 1024), 0), out=buf375)
        buf377 = buf337; del buf337  # reuse
        # Source Nodes: [bd_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf376, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf373, (16, 64, 1024), (64, 1, 1024), 0), out=buf377)
        buf380 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_56, add_57, attn_prob_18, attn_score_9, bd_19], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf375, buf377, buf380, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_56, add_57, attn_prob_18, attn_prob_19, attn_score_9, bd_19], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf381 = aten.native_dropout(buf380, 0.1, True)
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = reinterpret_tensor(buf370, (16, 512, 64), (32768, 64, 1), 0); del buf370  # reuse
        # Source Nodes: [attn_vec_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf382, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf372, (16, 512, 64), (64, 1024, 1), 0), out=buf384)
        buf385 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf384, buf385, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf386 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_70, buf386, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_70
        buf387 = reinterpret_tensor(buf384, (1, 512, 1024), (524288, 1024, 1), 0); del buf384  # reuse
        # Source Nodes: [attn_out_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf385, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf386, (1, 1024, 1024), (0, 1024, 1), 0), out=buf387)
        # Source Nodes: [attn_out_28], Original ATen: [aten.native_dropout]
        buf388 = aten.native_dropout(reinterpret_tensor(buf387, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf389 = buf388[0]
        buf390 = buf388[1]
        del buf388
        buf394 = reinterpret_tensor(buf387, (512, 1, 1024), (1024, 1024, 1), 0); del buf387  # reuse
        buf395 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1009 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_29, output_73, output_74], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf389, buf369, primals_242, primals_243, buf394, buf395, buf1009, 512, 1024, grid=grid(512), stream=stream0)
        buf396 = reinterpret_tensor(buf357, (512, 4096), (4096, 1), 0); del buf357  # reuse
        # Source Nodes: [output_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_245, buf395, reinterpret_tensor(primals_244, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf396)
        del primals_245
        buf397 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf396, buf397, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_75, output_76], Original ATen: [aten.gelu, aten.native_dropout]
        buf398 = aten.native_dropout(buf397, 0.1, True)
        buf399 = buf398[0]
        buf400 = buf398[1]
        del buf398
        buf401 = reinterpret_tensor(buf389, (512, 1024), (1024, 1), 0); del buf389  # reuse
        # Source Nodes: [output_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_247, reinterpret_tensor(buf399, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_246, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf401)
        del primals_247
        # Source Nodes: [output_78], Original ATen: [aten.native_dropout]
        buf402 = aten.native_dropout(reinterpret_tensor(buf401, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf403 = buf402[0]
        buf404 = buf402[1]
        del buf402
        buf408 = reinterpret_tensor(buf401, (512, 1, 1024), (1024, 1024, 1), 0); del buf401  # reuse
        buf409 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1008 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59, cat_11, output_73], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf403, buf394, primals_242, primals_243, primals_248, primals_249, buf408, buf409, buf1008, 512, 1024, grid=grid(512), stream=stream0)
        del primals_243
        del primals_249
        buf410 = reinterpret_tensor(buf403, (1, 512, 1024), (524288, 1024, 1), 0); del buf403  # reuse
        # Source Nodes: [q_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_71, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf410)
        buf411 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_72, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf411)
        buf412 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_73, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf412)
        buf413 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_74, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf413)
        del primals_74
        buf414 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf416 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, add_61], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf410, primals_75, primals_76, buf414, buf416, 524288, grid=grid(524288), stream=stream0)
        del primals_75
        del primals_76
        buf415 = buf375; del buf375  # reuse
        # Source Nodes: [ac_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf414, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf411, (16, 64, 512), (64, 1, 1024), 0), out=buf415)
        buf417 = buf377; del buf377  # reuse
        # Source Nodes: [bd_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf416, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf413, (16, 64, 1024), (64, 1, 1024), 0), out=buf417)
        buf420 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, add_63, attn_prob_20, attn_score_10, bd_21], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf415, buf417, buf420, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_62, add_63, attn_prob_20, attn_prob_21, attn_score_10, bd_21], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf421 = aten.native_dropout(buf420, 0.1, True)
        buf422 = buf421[0]
        buf423 = buf421[1]
        del buf421
        buf424 = reinterpret_tensor(buf410, (16, 512, 64), (32768, 64, 1), 0); del buf410  # reuse
        # Source Nodes: [attn_vec_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf412, (16, 512, 64), (64, 1024, 1), 0), out=buf424)
        buf425 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf424, buf425, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf426 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_77, buf426, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_77
        buf427 = reinterpret_tensor(buf424, (1, 512, 1024), (524288, 1024, 1), 0); del buf424  # reuse
        # Source Nodes: [attn_out_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf425, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf426, (1, 1024, 1024), (0, 1024, 1), 0), out=buf427)
        # Source Nodes: [attn_out_31], Original ATen: [aten.native_dropout]
        buf428 = aten.native_dropout(reinterpret_tensor(buf427, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf429 = buf428[0]
        buf430 = buf428[1]
        del buf428
        buf434 = reinterpret_tensor(buf427, (512, 1, 1024), (1024, 1024, 1), 0); del buf427  # reuse
        buf435 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1007 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_32, output_81, output_82], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf429, buf409, primals_250, primals_251, buf434, buf435, buf1007, 512, 1024, grid=grid(512), stream=stream0)
        buf436 = reinterpret_tensor(buf397, (512, 4096), (4096, 1), 0); del buf397  # reuse
        # Source Nodes: [output_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_253, buf435, reinterpret_tensor(primals_252, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf436)
        del primals_253
        buf437 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_83], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf436, buf437, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_83, output_84], Original ATen: [aten.gelu, aten.native_dropout]
        buf438 = aten.native_dropout(buf437, 0.1, True)
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = reinterpret_tensor(buf429, (512, 1024), (1024, 1), 0); del buf429  # reuse
        # Source Nodes: [output_85], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_255, reinterpret_tensor(buf439, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_254, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf441)
        del primals_255
        # Source Nodes: [output_86], Original ATen: [aten.native_dropout]
        buf442 = aten.native_dropout(reinterpret_tensor(buf441, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf443 = buf442[0]
        buf444 = buf442[1]
        del buf442
        buf448 = reinterpret_tensor(buf441, (512, 1, 1024), (1024, 1024, 1), 0); del buf441  # reuse
        buf449 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1006 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_65, cat_12, output_81], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf443, buf434, primals_250, primals_251, primals_256, primals_257, buf448, buf449, buf1006, 512, 1024, grid=grid(512), stream=stream0)
        del primals_251
        del primals_257
        buf450 = reinterpret_tensor(buf443, (1, 512, 1024), (524288, 1024, 1), 0); del buf443  # reuse
        # Source Nodes: [q_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf449, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_78, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf450)
        buf451 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf449, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_79, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf451)
        buf452 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf449, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_80, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf452)
        buf453 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_81, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf453)
        del primals_81
        buf454 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf456 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, add_67], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf450, primals_82, primals_83, buf454, buf456, 524288, grid=grid(524288), stream=stream0)
        del primals_82
        del primals_83
        buf455 = buf415; del buf415  # reuse
        # Source Nodes: [ac_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf454, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf451, (16, 64, 512), (64, 1, 1024), 0), out=buf455)
        buf457 = buf417; del buf417  # reuse
        # Source Nodes: [bd_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf456, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf453, (16, 64, 1024), (64, 1, 1024), 0), out=buf457)
        buf460 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, add_69, attn_prob_22, attn_score_11, bd_23], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf455, buf457, buf460, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_68, add_69, attn_prob_22, attn_prob_23, attn_score_11, bd_23], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf461 = aten.native_dropout(buf460, 0.1, True)
        buf462 = buf461[0]
        buf463 = buf461[1]
        del buf461
        buf464 = reinterpret_tensor(buf450, (16, 512, 64), (32768, 64, 1), 0); del buf450  # reuse
        # Source Nodes: [attn_vec_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf452, (16, 512, 64), (64, 1024, 1), 0), out=buf464)
        buf465 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf464, buf465, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf466 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_84, buf466, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_84
        buf467 = reinterpret_tensor(buf464, (1, 512, 1024), (524288, 1024, 1), 0); del buf464  # reuse
        # Source Nodes: [attn_out_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf465, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf466, (1, 1024, 1024), (0, 1024, 1), 0), out=buf467)
        # Source Nodes: [attn_out_34], Original ATen: [aten.native_dropout]
        buf468 = aten.native_dropout(reinterpret_tensor(buf467, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf469 = buf468[0]
        buf470 = buf468[1]
        del buf468
        buf474 = reinterpret_tensor(buf467, (512, 1, 1024), (1024, 1024, 1), 0); del buf467  # reuse
        buf475 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1005 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_35, output_89, output_90], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf469, buf449, primals_258, primals_259, buf474, buf475, buf1005, 512, 1024, grid=grid(512), stream=stream0)
        buf476 = reinterpret_tensor(buf437, (512, 4096), (4096, 1), 0); del buf437  # reuse
        # Source Nodes: [output_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_261, buf475, reinterpret_tensor(primals_260, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf476)
        del primals_261
        buf477 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_91], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf476, buf477, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_91, output_92], Original ATen: [aten.gelu, aten.native_dropout]
        buf478 = aten.native_dropout(buf477, 0.1, True)
        buf479 = buf478[0]
        buf480 = buf478[1]
        del buf478
        buf481 = reinterpret_tensor(buf469, (512, 1024), (1024, 1), 0); del buf469  # reuse
        # Source Nodes: [output_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_263, reinterpret_tensor(buf479, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_262, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf481)
        del primals_263
        # Source Nodes: [output_94], Original ATen: [aten.native_dropout]
        buf482 = aten.native_dropout(reinterpret_tensor(buf481, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf483 = buf482[0]
        buf484 = buf482[1]
        del buf482
        buf488 = reinterpret_tensor(buf481, (512, 1, 1024), (1024, 1024, 1), 0); del buf481  # reuse
        buf489 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1004 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71, cat_13, output_89], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf483, buf474, primals_258, primals_259, primals_264, primals_265, buf488, buf489, buf1004, 512, 1024, grid=grid(512), stream=stream0)
        del primals_259
        del primals_265
        buf490 = reinterpret_tensor(buf483, (1, 512, 1024), (524288, 1024, 1), 0); del buf483  # reuse
        # Source Nodes: [q_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_85, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf490)
        buf491 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_86, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf491)
        buf492 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_87, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf492)
        buf493 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_88, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf493)
        del primals_88
        buf494 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf496 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_72, add_73], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf490, primals_89, primals_90, buf494, buf496, 524288, grid=grid(524288), stream=stream0)
        del primals_89
        del primals_90
        buf495 = buf455; del buf455  # reuse
        # Source Nodes: [ac_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf494, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf491, (16, 64, 512), (64, 1, 1024), 0), out=buf495)
        buf497 = buf457; del buf457  # reuse
        # Source Nodes: [bd_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf493, (16, 64, 1024), (64, 1, 1024), 0), out=buf497)
        buf500 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_74, add_75, attn_prob_24, attn_score_12, bd_25], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf495, buf497, buf500, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_74, add_75, attn_prob_24, attn_prob_25, attn_score_12, bd_25], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf501 = aten.native_dropout(buf500, 0.1, True)
        buf502 = buf501[0]
        buf503 = buf501[1]
        del buf501
        buf504 = reinterpret_tensor(buf490, (16, 512, 64), (32768, 64, 1), 0); del buf490  # reuse
        # Source Nodes: [attn_vec_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf492, (16, 512, 64), (64, 1024, 1), 0), out=buf504)
        buf505 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf504, buf505, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf506 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_91, buf506, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_91
        buf507 = reinterpret_tensor(buf504, (1, 512, 1024), (524288, 1024, 1), 0); del buf504  # reuse
        # Source Nodes: [attn_out_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf505, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf506, (1, 1024, 1024), (0, 1024, 1), 0), out=buf507)
        # Source Nodes: [attn_out_37], Original ATen: [aten.native_dropout]
        buf508 = aten.native_dropout(reinterpret_tensor(buf507, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf514 = reinterpret_tensor(buf507, (512, 1, 1024), (1024, 1024, 1), 0); del buf507  # reuse
        buf515 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1003 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_38, output_97, output_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf509, buf489, primals_266, primals_267, buf514, buf515, buf1003, 512, 1024, grid=grid(512), stream=stream0)
        buf516 = reinterpret_tensor(buf477, (512, 4096), (4096, 1), 0); del buf477  # reuse
        # Source Nodes: [output_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_269, buf515, reinterpret_tensor(primals_268, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf516)
        del primals_269
        buf517 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf516, buf517, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_100, output_99], Original ATen: [aten.gelu, aten.native_dropout]
        buf518 = aten.native_dropout(buf517, 0.1, True)
        buf519 = buf518[0]
        buf520 = buf518[1]
        del buf518
        buf521 = reinterpret_tensor(buf509, (512, 1024), (1024, 1), 0); del buf509  # reuse
        # Source Nodes: [output_101], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_271, reinterpret_tensor(buf519, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_270, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf521)
        del primals_271
        # Source Nodes: [output_102], Original ATen: [aten.native_dropout]
        buf522 = aten.native_dropout(reinterpret_tensor(buf521, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf528 = reinterpret_tensor(buf521, (512, 1, 1024), (1024, 1024, 1), 0); del buf521  # reuse
        buf529 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1002 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_77, cat_14, output_97], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf523, buf514, primals_266, primals_267, primals_272, primals_273, buf528, buf529, buf1002, 512, 1024, grid=grid(512), stream=stream0)
        del primals_267
        del primals_273
        buf530 = reinterpret_tensor(buf523, (1, 512, 1024), (524288, 1024, 1), 0); del buf523  # reuse
        # Source Nodes: [q_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf529, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_92, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf530)
        buf531 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf529, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_93, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf531)
        buf532 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf529, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_94, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf532)
        buf533 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_95, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf533)
        del primals_95
        buf534 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf536 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_78, add_79], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf530, primals_96, primals_97, buf534, buf536, 524288, grid=grid(524288), stream=stream0)
        del primals_96
        del primals_97
        buf535 = buf495; del buf495  # reuse
        # Source Nodes: [ac_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf534, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf531, (16, 64, 512), (64, 1, 1024), 0), out=buf535)
        buf537 = buf497; del buf497  # reuse
        # Source Nodes: [bd_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf536, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf533, (16, 64, 1024), (64, 1, 1024), 0), out=buf537)
        buf540 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_80, add_81, attn_prob_26, attn_score_13, bd_27], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf535, buf537, buf540, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_80, add_81, attn_prob_26, attn_prob_27, attn_score_13, bd_27], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf541 = aten.native_dropout(buf540, 0.1, True)
        buf542 = buf541[0]
        buf543 = buf541[1]
        del buf541
        buf544 = reinterpret_tensor(buf530, (16, 512, 64), (32768, 64, 1), 0); del buf530  # reuse
        # Source Nodes: [attn_vec_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf532, (16, 512, 64), (64, 1024, 1), 0), out=buf544)
        buf545 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf544, buf545, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf546 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_98, buf546, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_98
        buf547 = reinterpret_tensor(buf544, (1, 512, 1024), (524288, 1024, 1), 0); del buf544  # reuse
        # Source Nodes: [attn_out_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf545, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf546, (1, 1024, 1024), (0, 1024, 1), 0), out=buf547)
        # Source Nodes: [attn_out_40], Original ATen: [aten.native_dropout]
        buf548 = aten.native_dropout(reinterpret_tensor(buf547, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf549 = buf548[0]
        buf550 = buf548[1]
        del buf548
        buf554 = reinterpret_tensor(buf547, (512, 1, 1024), (1024, 1024, 1), 0); del buf547  # reuse
        buf555 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf1001 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_41, output_105, output_106], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf549, buf529, primals_274, primals_275, buf554, buf555, buf1001, 512, 1024, grid=grid(512), stream=stream0)
        buf556 = reinterpret_tensor(buf517, (512, 4096), (4096, 1), 0); del buf517  # reuse
        # Source Nodes: [output_106], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_277, buf555, reinterpret_tensor(primals_276, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf556)
        del primals_277
        buf557 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf556, buf557, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_107, output_108], Original ATen: [aten.gelu, aten.native_dropout]
        buf558 = aten.native_dropout(buf557, 0.1, True)
        buf559 = buf558[0]
        buf560 = buf558[1]
        del buf558
        buf561 = reinterpret_tensor(buf549, (512, 1024), (1024, 1), 0); del buf549  # reuse
        # Source Nodes: [output_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_279, reinterpret_tensor(buf559, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_278, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf561)
        del primals_279
        # Source Nodes: [output_110], Original ATen: [aten.native_dropout]
        buf562 = aten.native_dropout(reinterpret_tensor(buf561, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf563 = buf562[0]
        buf564 = buf562[1]
        del buf562
        buf568 = reinterpret_tensor(buf561, (512, 1, 1024), (1024, 1024, 1), 0); del buf561  # reuse
        buf569 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf1000 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_83, cat_15, output_105], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf563, buf554, primals_274, primals_275, primals_280, primals_281, buf568, buf569, buf1000, 512, 1024, grid=grid(512), stream=stream0)
        del primals_275
        del primals_281
        buf570 = reinterpret_tensor(buf563, (1, 512, 1024), (524288, 1024, 1), 0); del buf563  # reuse
        # Source Nodes: [q_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf569, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_99, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf570)
        buf571 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf569, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_100, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf571)
        buf572 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf569, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_101, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf572)
        buf573 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_102, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf573)
        del primals_102
        buf574 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf576 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_84, add_85], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf570, primals_103, primals_104, buf574, buf576, 524288, grid=grid(524288), stream=stream0)
        del primals_103
        del primals_104
        buf575 = buf535; del buf535  # reuse
        # Source Nodes: [ac_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf571, (16, 64, 512), (64, 1, 1024), 0), out=buf575)
        buf577 = buf537; del buf537  # reuse
        # Source Nodes: [bd_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf576, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf573, (16, 64, 1024), (64, 1, 1024), 0), out=buf577)
        buf580 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_86, add_87, attn_prob_28, attn_score_14, bd_29], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf575, buf577, buf580, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_86, add_87, attn_prob_28, attn_prob_29, attn_score_14, bd_29], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf581 = aten.native_dropout(buf580, 0.1, True)
        buf582 = buf581[0]
        buf583 = buf581[1]
        del buf581
        buf584 = reinterpret_tensor(buf570, (16, 512, 64), (32768, 64, 1), 0); del buf570  # reuse
        # Source Nodes: [attn_vec_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf572, (16, 512, 64), (64, 1024, 1), 0), out=buf584)
        buf585 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf584, buf585, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf586 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_105, buf586, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_105
        buf587 = reinterpret_tensor(buf584, (1, 512, 1024), (524288, 1024, 1), 0); del buf584  # reuse
        # Source Nodes: [attn_out_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf586, (1, 1024, 1024), (0, 1024, 1), 0), out=buf587)
        # Source Nodes: [attn_out_43], Original ATen: [aten.native_dropout]
        buf588 = aten.native_dropout(reinterpret_tensor(buf587, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf589 = buf588[0]
        buf590 = buf588[1]
        del buf588
        buf594 = reinterpret_tensor(buf587, (512, 1, 1024), (1024, 1024, 1), 0); del buf587  # reuse
        buf595 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf999 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_44, output_113, output_114], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf589, buf569, primals_282, primals_283, buf594, buf595, buf999, 512, 1024, grid=grid(512), stream=stream0)
        buf596 = reinterpret_tensor(buf557, (512, 4096), (4096, 1), 0); del buf557  # reuse
        # Source Nodes: [output_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_285, buf595, reinterpret_tensor(primals_284, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf596)
        del primals_285
        buf597 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_115], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf596, buf597, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_115, output_116], Original ATen: [aten.gelu, aten.native_dropout]
        buf598 = aten.native_dropout(buf597, 0.1, True)
        buf599 = buf598[0]
        buf600 = buf598[1]
        del buf598
        buf601 = reinterpret_tensor(buf589, (512, 1024), (1024, 1), 0); del buf589  # reuse
        # Source Nodes: [output_117], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_287, reinterpret_tensor(buf599, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_286, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf601)
        del primals_287
        # Source Nodes: [output_118], Original ATen: [aten.native_dropout]
        buf602 = aten.native_dropout(reinterpret_tensor(buf601, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf603 = buf602[0]
        buf604 = buf602[1]
        del buf602
        buf608 = reinterpret_tensor(buf601, (512, 1, 1024), (1024, 1024, 1), 0); del buf601  # reuse
        buf609 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf998 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_89, cat_16, output_113], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf603, buf594, primals_282, primals_283, primals_288, primals_289, buf608, buf609, buf998, 512, 1024, grid=grid(512), stream=stream0)
        del primals_283
        del primals_289
        buf610 = reinterpret_tensor(buf603, (1, 512, 1024), (524288, 1024, 1), 0); del buf603  # reuse
        # Source Nodes: [q_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf609, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_106, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf610)
        buf611 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf609, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_107, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf611)
        buf612 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf609, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_108, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf612)
        buf613 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_109, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf613)
        del primals_109
        buf614 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf616 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_90, add_91], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf610, primals_110, primals_111, buf614, buf616, 524288, grid=grid(524288), stream=stream0)
        del primals_110
        del primals_111
        buf615 = buf575; del buf575  # reuse
        # Source Nodes: [ac_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf614, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf611, (16, 64, 512), (64, 1, 1024), 0), out=buf615)
        buf617 = buf577; del buf577  # reuse
        # Source Nodes: [bd_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf616, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf613, (16, 64, 1024), (64, 1, 1024), 0), out=buf617)
        buf620 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_92, add_93, attn_prob_30, attn_score_15, bd_31], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf615, buf617, buf620, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_92, add_93, attn_prob_30, attn_prob_31, attn_score_15, bd_31], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf621 = aten.native_dropout(buf620, 0.1, True)
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf624 = reinterpret_tensor(buf610, (16, 512, 64), (32768, 64, 1), 0); del buf610  # reuse
        # Source Nodes: [attn_vec_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf622, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf612, (16, 512, 64), (64, 1024, 1), 0), out=buf624)
        buf625 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf624, buf625, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf626 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_112, buf626, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_112
        buf627 = reinterpret_tensor(buf624, (1, 512, 1024), (524288, 1024, 1), 0); del buf624  # reuse
        # Source Nodes: [attn_out_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf625, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf626, (1, 1024, 1024), (0, 1024, 1), 0), out=buf627)
        # Source Nodes: [attn_out_46], Original ATen: [aten.native_dropout]
        buf628 = aten.native_dropout(reinterpret_tensor(buf627, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf629 = buf628[0]
        buf630 = buf628[1]
        del buf628
        buf634 = reinterpret_tensor(buf627, (512, 1, 1024), (1024, 1024, 1), 0); del buf627  # reuse
        buf635 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf997 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_47, output_121, output_122], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf629, buf609, primals_290, primals_291, buf634, buf635, buf997, 512, 1024, grid=grid(512), stream=stream0)
        buf636 = reinterpret_tensor(buf597, (512, 4096), (4096, 1), 0); del buf597  # reuse
        # Source Nodes: [output_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_293, buf635, reinterpret_tensor(primals_292, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf636)
        del primals_293
        buf637 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf636, buf637, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_123, output_124], Original ATen: [aten.gelu, aten.native_dropout]
        buf638 = aten.native_dropout(buf637, 0.1, True)
        buf639 = buf638[0]
        buf640 = buf638[1]
        del buf638
        buf641 = reinterpret_tensor(buf629, (512, 1024), (1024, 1), 0); del buf629  # reuse
        # Source Nodes: [output_125], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_295, reinterpret_tensor(buf639, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_294, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf641)
        del primals_295
        # Source Nodes: [output_126], Original ATen: [aten.native_dropout]
        buf642 = aten.native_dropout(reinterpret_tensor(buf641, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf643 = buf642[0]
        buf644 = buf642[1]
        del buf642
        buf648 = reinterpret_tensor(buf641, (512, 1, 1024), (1024, 1024, 1), 0); del buf641  # reuse
        buf649 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf996 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_95, cat_17, output_121], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf643, buf634, primals_290, primals_291, primals_296, primals_297, buf648, buf649, buf996, 512, 1024, grid=grid(512), stream=stream0)
        del primals_291
        del primals_297
        buf650 = reinterpret_tensor(buf643, (1, 512, 1024), (524288, 1024, 1), 0); del buf643  # reuse
        # Source Nodes: [q_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf649, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_113, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf650)
        buf651 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf649, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_114, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf651)
        buf652 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf649, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_115, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf652)
        buf653 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_116, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf653)
        del primals_116
        buf654 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf656 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_96, add_97], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf650, primals_117, primals_118, buf654, buf656, 524288, grid=grid(524288), stream=stream0)
        del primals_117
        del primals_118
        buf655 = buf615; del buf615  # reuse
        # Source Nodes: [ac_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf654, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf651, (16, 64, 512), (64, 1, 1024), 0), out=buf655)
        buf657 = buf617; del buf617  # reuse
        # Source Nodes: [bd_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf656, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf653, (16, 64, 1024), (64, 1, 1024), 0), out=buf657)
        buf660 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_98, add_99, attn_prob_32, attn_score_16, bd_33], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf655, buf657, buf660, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_98, add_99, attn_prob_32, attn_prob_33, attn_score_16, bd_33], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf661 = aten.native_dropout(buf660, 0.1, True)
        buf662 = buf661[0]
        buf663 = buf661[1]
        del buf661
        buf664 = reinterpret_tensor(buf650, (16, 512, 64), (32768, 64, 1), 0); del buf650  # reuse
        # Source Nodes: [attn_vec_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf662, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf652, (16, 512, 64), (64, 1024, 1), 0), out=buf664)
        buf665 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf664, buf665, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf666 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_119, buf666, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_119
        buf667 = reinterpret_tensor(buf664, (1, 512, 1024), (524288, 1024, 1), 0); del buf664  # reuse
        # Source Nodes: [attn_out_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf665, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf666, (1, 1024, 1024), (0, 1024, 1), 0), out=buf667)
        # Source Nodes: [attn_out_49], Original ATen: [aten.native_dropout]
        buf668 = aten.native_dropout(reinterpret_tensor(buf667, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf669 = buf668[0]
        buf670 = buf668[1]
        del buf668
        buf674 = reinterpret_tensor(buf667, (512, 1, 1024), (1024, 1024, 1), 0); del buf667  # reuse
        buf675 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf995 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_50, output_129, output_130], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf669, buf649, primals_298, primals_299, buf674, buf675, buf995, 512, 1024, grid=grid(512), stream=stream0)
        buf676 = reinterpret_tensor(buf637, (512, 4096), (4096, 1), 0); del buf637  # reuse
        # Source Nodes: [output_130], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_301, buf675, reinterpret_tensor(primals_300, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf676)
        del primals_301
        buf677 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_131], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf676, buf677, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_131, output_132], Original ATen: [aten.gelu, aten.native_dropout]
        buf678 = aten.native_dropout(buf677, 0.1, True)
        buf679 = buf678[0]
        buf680 = buf678[1]
        del buf678
        buf681 = reinterpret_tensor(buf669, (512, 1024), (1024, 1), 0); del buf669  # reuse
        # Source Nodes: [output_133], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_303, reinterpret_tensor(buf679, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_302, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf681)
        del primals_303
        # Source Nodes: [output_134], Original ATen: [aten.native_dropout]
        buf682 = aten.native_dropout(reinterpret_tensor(buf681, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf683 = buf682[0]
        buf684 = buf682[1]
        del buf682
        buf688 = reinterpret_tensor(buf681, (512, 1, 1024), (1024, 1024, 1), 0); del buf681  # reuse
        buf689 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf994 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_101, cat_18, output_129], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf683, buf674, primals_298, primals_299, primals_304, primals_305, buf688, buf689, buf994, 512, 1024, grid=grid(512), stream=stream0)
        del primals_299
        del primals_305
        buf690 = reinterpret_tensor(buf683, (1, 512, 1024), (524288, 1024, 1), 0); del buf683  # reuse
        # Source Nodes: [q_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf689, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_120, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf690)
        buf691 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf689, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_121, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf691)
        buf692 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf689, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_122, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf692)
        buf693 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_123, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf693)
        del primals_123
        buf694 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf696 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_102, add_103], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf690, primals_124, primals_125, buf694, buf696, 524288, grid=grid(524288), stream=stream0)
        del primals_124
        del primals_125
        buf695 = buf655; del buf655  # reuse
        # Source Nodes: [ac_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf694, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf691, (16, 64, 512), (64, 1, 1024), 0), out=buf695)
        buf697 = buf657; del buf657  # reuse
        # Source Nodes: [bd_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf696, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf693, (16, 64, 1024), (64, 1, 1024), 0), out=buf697)
        buf700 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_104, add_105, attn_prob_34, attn_score_17, bd_35], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf695, buf697, buf700, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_104, add_105, attn_prob_34, attn_prob_35, attn_score_17, bd_35], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf701 = aten.native_dropout(buf700, 0.1, True)
        buf702 = buf701[0]
        buf703 = buf701[1]
        del buf701
        buf704 = reinterpret_tensor(buf690, (16, 512, 64), (32768, 64, 1), 0); del buf690  # reuse
        # Source Nodes: [attn_vec_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf692, (16, 512, 64), (64, 1024, 1), 0), out=buf704)
        buf705 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf704, buf705, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf706 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_126, buf706, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_126
        buf707 = reinterpret_tensor(buf704, (1, 512, 1024), (524288, 1024, 1), 0); del buf704  # reuse
        # Source Nodes: [attn_out_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf705, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf706, (1, 1024, 1024), (0, 1024, 1), 0), out=buf707)
        # Source Nodes: [attn_out_52], Original ATen: [aten.native_dropout]
        buf708 = aten.native_dropout(reinterpret_tensor(buf707, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf709 = buf708[0]
        buf710 = buf708[1]
        del buf708
        buf714 = reinterpret_tensor(buf707, (512, 1, 1024), (1024, 1024, 1), 0); del buf707  # reuse
        buf715 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf993 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_53, output_137, output_138], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf709, buf689, primals_306, primals_307, buf714, buf715, buf993, 512, 1024, grid=grid(512), stream=stream0)
        buf716 = reinterpret_tensor(buf677, (512, 4096), (4096, 1), 0); del buf677  # reuse
        # Source Nodes: [output_138], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_309, buf715, reinterpret_tensor(primals_308, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf716)
        del primals_309
        buf717 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_139], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf716, buf717, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_139, output_140], Original ATen: [aten.gelu, aten.native_dropout]
        buf718 = aten.native_dropout(buf717, 0.1, True)
        buf719 = buf718[0]
        buf720 = buf718[1]
        del buf718
        buf721 = reinterpret_tensor(buf709, (512, 1024), (1024, 1), 0); del buf709  # reuse
        # Source Nodes: [output_141], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_311, reinterpret_tensor(buf719, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_310, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf721)
        del primals_311
        # Source Nodes: [output_142], Original ATen: [aten.native_dropout]
        buf722 = aten.native_dropout(reinterpret_tensor(buf721, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf723 = buf722[0]
        buf724 = buf722[1]
        del buf722
        buf728 = reinterpret_tensor(buf721, (512, 1, 1024), (1024, 1024, 1), 0); del buf721  # reuse
        buf729 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf992 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_107, cat_19, output_137], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf723, buf714, primals_306, primals_307, primals_312, primals_313, buf728, buf729, buf992, 512, 1024, grid=grid(512), stream=stream0)
        del primals_307
        del primals_313
        buf730 = reinterpret_tensor(buf723, (1, 512, 1024), (524288, 1024, 1), 0); del buf723  # reuse
        # Source Nodes: [q_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_127, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf730)
        buf731 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_128, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf731)
        buf732 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_129, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf732)
        buf733 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_130, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf733)
        del primals_130
        buf734 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf736 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_108, add_109], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf730, primals_131, primals_132, buf734, buf736, 524288, grid=grid(524288), stream=stream0)
        del primals_131
        del primals_132
        buf735 = buf695; del buf695  # reuse
        # Source Nodes: [ac_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf734, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf731, (16, 64, 512), (64, 1, 1024), 0), out=buf735)
        buf737 = buf697; del buf697  # reuse
        # Source Nodes: [bd_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf736, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf733, (16, 64, 1024), (64, 1, 1024), 0), out=buf737)
        buf740 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_110, add_111, attn_prob_36, attn_score_18, bd_37], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf735, buf737, buf740, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_110, add_111, attn_prob_36, attn_prob_37, attn_score_18, bd_37], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf741 = aten.native_dropout(buf740, 0.1, True)
        buf742 = buf741[0]
        buf743 = buf741[1]
        del buf741
        buf744 = reinterpret_tensor(buf730, (16, 512, 64), (32768, 64, 1), 0); del buf730  # reuse
        # Source Nodes: [attn_vec_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf742, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf732, (16, 512, 64), (64, 1024, 1), 0), out=buf744)
        buf745 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf744, buf745, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf746 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_133, buf746, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_133
        buf747 = reinterpret_tensor(buf744, (1, 512, 1024), (524288, 1024, 1), 0); del buf744  # reuse
        # Source Nodes: [attn_out_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf745, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf746, (1, 1024, 1024), (0, 1024, 1), 0), out=buf747)
        # Source Nodes: [attn_out_55], Original ATen: [aten.native_dropout]
        buf748 = aten.native_dropout(reinterpret_tensor(buf747, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf749 = buf748[0]
        buf750 = buf748[1]
        del buf748
        buf754 = reinterpret_tensor(buf747, (512, 1, 1024), (1024, 1024, 1), 0); del buf747  # reuse
        buf755 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf991 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_56, output_145, output_146], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf749, buf729, primals_314, primals_315, buf754, buf755, buf991, 512, 1024, grid=grid(512), stream=stream0)
        buf756 = reinterpret_tensor(buf717, (512, 4096), (4096, 1), 0); del buf717  # reuse
        # Source Nodes: [output_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_317, buf755, reinterpret_tensor(primals_316, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf756)
        del primals_317
        buf757 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_147], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf756, buf757, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_147, output_148], Original ATen: [aten.gelu, aten.native_dropout]
        buf758 = aten.native_dropout(buf757, 0.1, True)
        buf759 = buf758[0]
        buf760 = buf758[1]
        del buf758
        buf761 = reinterpret_tensor(buf749, (512, 1024), (1024, 1), 0); del buf749  # reuse
        # Source Nodes: [output_149], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_319, reinterpret_tensor(buf759, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_318, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf761)
        del primals_319
        # Source Nodes: [output_150], Original ATen: [aten.native_dropout]
        buf762 = aten.native_dropout(reinterpret_tensor(buf761, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf763 = buf762[0]
        buf764 = buf762[1]
        del buf762
        buf768 = reinterpret_tensor(buf761, (512, 1, 1024), (1024, 1024, 1), 0); del buf761  # reuse
        buf769 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf990 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_113, cat_20, output_145], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf763, buf754, primals_314, primals_315, primals_320, primals_321, buf768, buf769, buf990, 512, 1024, grid=grid(512), stream=stream0)
        del primals_315
        del primals_321
        buf770 = reinterpret_tensor(buf763, (1, 512, 1024), (524288, 1024, 1), 0); del buf763  # reuse
        # Source Nodes: [q_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf769, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_134, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf770)
        buf771 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf769, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_135, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf771)
        buf772 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf769, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_136, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf772)
        buf773 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_137, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf773)
        del primals_137
        buf774 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf776 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_114, add_115], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf770, primals_138, primals_139, buf774, buf776, 524288, grid=grid(524288), stream=stream0)
        del primals_138
        del primals_139
        buf775 = buf735; del buf735  # reuse
        # Source Nodes: [ac_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf774, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf771, (16, 64, 512), (64, 1, 1024), 0), out=buf775)
        buf777 = buf737; del buf737  # reuse
        # Source Nodes: [bd_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf776, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf773, (16, 64, 1024), (64, 1, 1024), 0), out=buf777)
        buf780 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_116, add_117, attn_prob_38, attn_score_19, bd_39], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf775, buf777, buf780, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_116, add_117, attn_prob_38, attn_prob_39, attn_score_19, bd_39], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf781 = aten.native_dropout(buf780, 0.1, True)
        buf782 = buf781[0]
        buf783 = buf781[1]
        del buf781
        buf784 = reinterpret_tensor(buf770, (16, 512, 64), (32768, 64, 1), 0); del buf770  # reuse
        # Source Nodes: [attn_vec_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf782, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf772, (16, 512, 64), (64, 1024, 1), 0), out=buf784)
        buf785 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf784, buf785, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf786 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_140, buf786, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_140
        buf787 = reinterpret_tensor(buf784, (1, 512, 1024), (524288, 1024, 1), 0); del buf784  # reuse
        # Source Nodes: [attn_out_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf785, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf786, (1, 1024, 1024), (0, 1024, 1), 0), out=buf787)
        # Source Nodes: [attn_out_58], Original ATen: [aten.native_dropout]
        buf788 = aten.native_dropout(reinterpret_tensor(buf787, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf789 = buf788[0]
        buf790 = buf788[1]
        del buf788
        buf794 = reinterpret_tensor(buf787, (512, 1, 1024), (1024, 1024, 1), 0); del buf787  # reuse
        buf795 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf989 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_59, output_153, output_154], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf789, buf769, primals_322, primals_323, buf794, buf795, buf989, 512, 1024, grid=grid(512), stream=stream0)
        buf796 = reinterpret_tensor(buf757, (512, 4096), (4096, 1), 0); del buf757  # reuse
        # Source Nodes: [output_154], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_325, buf795, reinterpret_tensor(primals_324, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf796)
        del primals_325
        buf797 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_155], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf796, buf797, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_155, output_156], Original ATen: [aten.gelu, aten.native_dropout]
        buf798 = aten.native_dropout(buf797, 0.1, True)
        buf799 = buf798[0]
        buf800 = buf798[1]
        del buf798
        buf801 = reinterpret_tensor(buf789, (512, 1024), (1024, 1), 0); del buf789  # reuse
        # Source Nodes: [output_157], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_327, reinterpret_tensor(buf799, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_326, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf801)
        del primals_327
        # Source Nodes: [output_158], Original ATen: [aten.native_dropout]
        buf802 = aten.native_dropout(reinterpret_tensor(buf801, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf803 = buf802[0]
        buf804 = buf802[1]
        del buf802
        buf808 = reinterpret_tensor(buf801, (512, 1, 1024), (1024, 1024, 1), 0); del buf801  # reuse
        buf809 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf988 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_119, cat_21, output_153], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf803, buf794, primals_322, primals_323, primals_328, primals_329, buf808, buf809, buf988, 512, 1024, grid=grid(512), stream=stream0)
        del primals_323
        del primals_329
        buf810 = reinterpret_tensor(buf803, (1, 512, 1024), (524288, 1024, 1), 0); del buf803  # reuse
        # Source Nodes: [q_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf809, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_141, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf810)
        buf811 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf809, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_142, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf811)
        buf812 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf809, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_143, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf812)
        buf813 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_144, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf813)
        del primals_144
        buf814 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf816 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_120, add_121], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf810, primals_145, primals_146, buf814, buf816, 524288, grid=grid(524288), stream=stream0)
        del primals_145
        del primals_146
        buf815 = buf775; del buf775  # reuse
        # Source Nodes: [ac_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf814, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf811, (16, 64, 512), (64, 1, 1024), 0), out=buf815)
        buf817 = buf777; del buf777  # reuse
        # Source Nodes: [bd_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf816, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf813, (16, 64, 1024), (64, 1, 1024), 0), out=buf817)
        buf820 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_122, add_123, attn_prob_40, attn_score_20, bd_41], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf815, buf817, buf820, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_122, add_123, attn_prob_40, attn_prob_41, attn_score_20, bd_41], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf821 = aten.native_dropout(buf820, 0.1, True)
        buf822 = buf821[0]
        buf823 = buf821[1]
        del buf821
        buf824 = reinterpret_tensor(buf810, (16, 512, 64), (32768, 64, 1), 0); del buf810  # reuse
        # Source Nodes: [attn_vec_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf822, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf812, (16, 512, 64), (64, 1024, 1), 0), out=buf824)
        buf825 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf824, buf825, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf826 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_147, buf826, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_147
        buf827 = reinterpret_tensor(buf824, (1, 512, 1024), (524288, 1024, 1), 0); del buf824  # reuse
        # Source Nodes: [attn_out_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf825, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf826, (1, 1024, 1024), (0, 1024, 1), 0), out=buf827)
        # Source Nodes: [attn_out_61], Original ATen: [aten.native_dropout]
        buf828 = aten.native_dropout(reinterpret_tensor(buf827, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf829 = buf828[0]
        buf830 = buf828[1]
        del buf828
        buf834 = reinterpret_tensor(buf827, (512, 1, 1024), (1024, 1024, 1), 0); del buf827  # reuse
        buf835 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf987 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_62, output_161, output_162], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf829, buf809, primals_330, primals_331, buf834, buf835, buf987, 512, 1024, grid=grid(512), stream=stream0)
        buf836 = reinterpret_tensor(buf797, (512, 4096), (4096, 1), 0); del buf797  # reuse
        # Source Nodes: [output_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_333, buf835, reinterpret_tensor(primals_332, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf836)
        del primals_333
        buf837 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf836, buf837, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_163, output_164], Original ATen: [aten.gelu, aten.native_dropout]
        buf838 = aten.native_dropout(buf837, 0.1, True)
        buf839 = buf838[0]
        buf840 = buf838[1]
        del buf838
        buf841 = reinterpret_tensor(buf829, (512, 1024), (1024, 1), 0); del buf829  # reuse
        # Source Nodes: [output_165], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_335, reinterpret_tensor(buf839, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_334, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf841)
        del primals_335
        # Source Nodes: [output_166], Original ATen: [aten.native_dropout]
        buf842 = aten.native_dropout(reinterpret_tensor(buf841, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf843 = buf842[0]
        buf844 = buf842[1]
        del buf842
        buf848 = reinterpret_tensor(buf841, (512, 1, 1024), (1024, 1024, 1), 0); del buf841  # reuse
        buf849 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf986 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_125, cat_22, output_161], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf843, buf834, primals_330, primals_331, primals_336, primals_337, buf848, buf849, buf986, 512, 1024, grid=grid(512), stream=stream0)
        del primals_331
        del primals_337
        buf850 = reinterpret_tensor(buf843, (1, 512, 1024), (524288, 1024, 1), 0); del buf843  # reuse
        # Source Nodes: [q_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf849, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_148, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf850)
        buf851 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf849, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_149, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf851)
        buf852 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf849, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_150, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf852)
        buf853 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_151, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf853)
        del primals_151
        buf854 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf856 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, add_127], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf850, primals_152, primals_153, buf854, buf856, 524288, grid=grid(524288), stream=stream0)
        del primals_152
        del primals_153
        buf855 = buf815; del buf815  # reuse
        # Source Nodes: [ac_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf854, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf851, (16, 64, 512), (64, 1, 1024), 0), out=buf855)
        buf857 = buf817; del buf817  # reuse
        # Source Nodes: [bd_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf856, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf853, (16, 64, 1024), (64, 1, 1024), 0), out=buf857)
        buf860 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_128, add_129, attn_prob_42, attn_score_21, bd_43], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf855, buf857, buf860, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_128, add_129, attn_prob_42, attn_prob_43, attn_score_21, bd_43], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf861 = aten.native_dropout(buf860, 0.1, True)
        buf862 = buf861[0]
        buf863 = buf861[1]
        del buf861
        buf864 = reinterpret_tensor(buf850, (16, 512, 64), (32768, 64, 1), 0); del buf850  # reuse
        # Source Nodes: [attn_vec_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf862, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf852, (16, 512, 64), (64, 1024, 1), 0), out=buf864)
        buf865 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf864, buf865, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf866 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_154, buf866, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_154
        buf867 = reinterpret_tensor(buf864, (1, 512, 1024), (524288, 1024, 1), 0); del buf864  # reuse
        # Source Nodes: [attn_out_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf865, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf866, (1, 1024, 1024), (0, 1024, 1), 0), out=buf867)
        # Source Nodes: [attn_out_64], Original ATen: [aten.native_dropout]
        buf868 = aten.native_dropout(reinterpret_tensor(buf867, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf869 = buf868[0]
        buf870 = buf868[1]
        del buf868
        buf874 = reinterpret_tensor(buf867, (512, 1, 1024), (1024, 1024, 1), 0); del buf867  # reuse
        buf875 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf985 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_65, output_169, output_170], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf869, buf849, primals_338, primals_339, buf874, buf875, buf985, 512, 1024, grid=grid(512), stream=stream0)
        buf876 = reinterpret_tensor(buf837, (512, 4096), (4096, 1), 0); del buf837  # reuse
        # Source Nodes: [output_170], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_341, buf875, reinterpret_tensor(primals_340, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf876)
        del primals_341
        buf877 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_171], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf876, buf877, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_171, output_172], Original ATen: [aten.gelu, aten.native_dropout]
        buf878 = aten.native_dropout(buf877, 0.1, True)
        buf879 = buf878[0]
        buf880 = buf878[1]
        del buf878
        buf881 = reinterpret_tensor(buf869, (512, 1024), (1024, 1), 0); del buf869  # reuse
        # Source Nodes: [output_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_343, reinterpret_tensor(buf879, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_342, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf881)
        del primals_343
        # Source Nodes: [output_174], Original ATen: [aten.native_dropout]
        buf882 = aten.native_dropout(reinterpret_tensor(buf881, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf883 = buf882[0]
        buf884 = buf882[1]
        del buf882
        buf888 = reinterpret_tensor(buf881, (512, 1, 1024), (1024, 1024, 1), 0); del buf881  # reuse
        buf889 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf984 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_131, cat_23, output_169], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf883, buf874, primals_338, primals_339, primals_344, primals_345, buf888, buf889, buf984, 512, 1024, grid=grid(512), stream=stream0)
        del primals_339
        del primals_345
        buf890 = reinterpret_tensor(buf883, (1, 512, 1024), (524288, 1024, 1), 0); del buf883  # reuse
        # Source Nodes: [q_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf889, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_155, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf890)
        buf891 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf889, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_156, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf891)
        buf892 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf889, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_157, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf892)
        buf893 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_158, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf893)
        del primals_158
        buf894 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf896 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_132, add_133], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf890, primals_159, primals_160, buf894, buf896, 524288, grid=grid(524288), stream=stream0)
        del primals_159
        del primals_160
        buf895 = buf855; del buf855  # reuse
        # Source Nodes: [ac_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf894, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf891, (16, 64, 512), (64, 1, 1024), 0), out=buf895)
        buf897 = buf857; del buf857  # reuse
        # Source Nodes: [bd_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf896, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf893, (16, 64, 1024), (64, 1, 1024), 0), out=buf897)
        buf900 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_134, add_135, attn_prob_44, attn_score_22, bd_45], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf895, buf897, buf900, 8192, 512, grid=grid(8192), stream=stream0)
        # Source Nodes: [add_134, add_135, attn_prob_44, attn_prob_45, attn_score_22, bd_45], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf901 = aten.native_dropout(buf900, 0.1, True)
        buf902 = buf901[0]
        buf903 = buf901[1]
        del buf901
        buf904 = reinterpret_tensor(buf890, (16, 512, 64), (32768, 64, 1), 0); del buf890  # reuse
        # Source Nodes: [attn_vec_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf902, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf892, (16, 512, 64), (64, 1024, 1), 0), out=buf904)
        buf905 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf904, buf905, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf906 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_161, buf906, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_161
        buf907 = reinterpret_tensor(buf904, (1, 512, 1024), (524288, 1024, 1), 0); del buf904  # reuse
        # Source Nodes: [attn_out_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf905, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf906, (1, 1024, 1024), (0, 1024, 1), 0), out=buf907)
        # Source Nodes: [attn_out_67], Original ATen: [aten.native_dropout]
        buf908 = aten.native_dropout(reinterpret_tensor(buf907, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf909 = buf908[0]
        buf910 = buf908[1]
        del buf908
        buf914 = reinterpret_tensor(buf907, (512, 1, 1024), (1024, 1024, 1), 0); del buf907  # reuse
        buf915 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf983 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_68, output_177, output_178], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf909, buf889, primals_346, primals_347, buf914, buf915, buf983, 512, 1024, grid=grid(512), stream=stream0)
        buf916 = reinterpret_tensor(buf877, (512, 4096), (4096, 1), 0); del buf877  # reuse
        # Source Nodes: [output_178], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_349, buf915, reinterpret_tensor(primals_348, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf916)
        del primals_349
        buf917 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_179], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf916, buf917, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_179, output_180], Original ATen: [aten.gelu, aten.native_dropout]
        buf918 = aten.native_dropout(buf917, 0.1, True)
        buf919 = buf918[0]
        buf920 = buf918[1]
        del buf918
        buf921 = reinterpret_tensor(buf909, (512, 1024), (1024, 1), 0); del buf909  # reuse
        # Source Nodes: [output_181], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_351, reinterpret_tensor(buf919, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_350, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf921)
        del primals_351
        # Source Nodes: [output_182], Original ATen: [aten.native_dropout]
        buf922 = aten.native_dropout(reinterpret_tensor(buf921, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf923 = buf922[0]
        buf924 = buf922[1]
        del buf922
        buf928 = reinterpret_tensor(buf921, (512, 1, 1024), (1024, 1024, 1), 0); del buf921  # reuse
        buf929 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        buf982 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_137, cat_24, output_177], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf923, buf914, primals_346, primals_347, primals_352, primals_353, buf928, buf929, buf982, 512, 1024, grid=grid(512), stream=stream0)
        del primals_347
        del primals_353
        buf930 = reinterpret_tensor(buf923, (1, 512, 1024), (524288, 1024, 1), 0); del buf923  # reuse
        # Source Nodes: [q_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf929, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_162, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf930)
        buf931 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf929, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_163, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf931)
        buf932 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf929, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(primals_164, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf932)
        buf933 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(primals_165, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf933)
        del primals_165
        buf934 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf936 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_138, add_139], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf930, primals_166, primals_167, buf934, buf936, 524288, grid=grid(524288), stream=stream0)
        del primals_166
        del primals_167
        buf935 = buf895; del buf895  # reuse
        # Source Nodes: [ac_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf934, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf931, (16, 64, 512), (64, 1, 1024), 0), out=buf935)
        buf937 = buf897; del buf897  # reuse
        # Source Nodes: [bd_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf936, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf933, (16, 64, 1024), (64, 1, 1024), 0), out=buf937)
        buf940 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_140, add_141, attn_prob_46, attn_score_23, bd_47], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_4.run(buf935, buf937, buf940, 8192, 512, grid=grid(8192), stream=stream0)
        del buf935
        del buf937
        # Source Nodes: [add_140, add_141, attn_prob_46, attn_prob_47, attn_score_23, bd_47], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
        buf941 = aten.native_dropout(buf940, 0.1, True)
        buf942 = buf941[0]
        buf943 = buf941[1]
        del buf941
        buf944 = reinterpret_tensor(buf930, (16, 512, 64), (32768, 64, 1), 0); del buf930  # reuse
        # Source Nodes: [attn_vec_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf942, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf932, (16, 512, 64), (64, 1024, 1), 0), out=buf944)
        buf945 = empty((512, 64, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf944, buf945, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf946 = empty((64, 16, 1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(primals_168, buf946, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del primals_168
        buf947 = reinterpret_tensor(buf944, (1, 512, 1024), (524288, 1024, 1), 0); del buf944  # reuse
        # Source Nodes: [attn_out_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf945, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf946, (1, 1024, 1024), (0, 1024, 1), 0), out=buf947)
        # Source Nodes: [attn_out_70], Original ATen: [aten.native_dropout]
        buf948 = aten.native_dropout(reinterpret_tensor(buf947, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf949 = buf948[0]
        buf950 = buf948[1]
        del buf948
        buf954 = reinterpret_tensor(buf947, (512, 1, 1024), (1024, 1024, 1), 0); del buf947  # reuse
        buf955 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf981 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_out_71, output_185, output_186], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf949, buf929, primals_354, primals_355, buf954, buf955, buf981, 512, 1024, grid=grid(512), stream=stream0)
        buf956 = reinterpret_tensor(buf917, (512, 4096), (4096, 1), 0); del buf917  # reuse
        # Source Nodes: [output_186], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_357, buf955, reinterpret_tensor(primals_356, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf956)
        del primals_357
        buf957 = empty_strided((512, 1, 4096), (4096, 2097152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [output_187], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf956, buf957, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [output_187, output_188], Original ATen: [aten.gelu, aten.native_dropout]
        buf958 = aten.native_dropout(buf957, 0.1, True)
        del buf957
        buf959 = buf958[0]
        buf960 = buf958[1]
        del buf958
        buf961 = reinterpret_tensor(buf949, (512, 1024), (1024, 1), 0); del buf949  # reuse
        # Source Nodes: [output_189], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_359, reinterpret_tensor(buf959, (512, 4096), (4096, 1), 0), reinterpret_tensor(primals_358, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf961)
        del primals_359
        # Source Nodes: [output_190], Original ATen: [aten.native_dropout]
        buf962 = aten.native_dropout(reinterpret_tensor(buf961, (512, 1, 1024), (1024, 1024, 1), 0), 0.1, True)
        buf963 = buf962[0]
        buf964 = buf962[1]
        del buf962
        buf968 = reinterpret_tensor(buf961, (512, 1, 1024), (1024, 1024, 1), 0); del buf961  # reuse
        buf969 = empty_strided((512, 1, 1024), (1024, 524288, 1), device='cuda', dtype=torch.float32)
        buf980 = empty((512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_143, output_185, output_h_96], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf963, buf954, primals_354, primals_355, primals_360, primals_361, buf968, buf969, buf980, 512, 1024, grid=grid(512), stream=stream0)
        del buf963
        del primals_355
        del primals_361
        # Source Nodes: [output_192, output_h_96], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf970 = aten.native_dropout(buf969, 0.1, True)
        del buf969
        buf971 = buf970[0]
        buf972 = buf970[1]
        del buf970
        buf973 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_363, reinterpret_tensor(buf971, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_362, (1024, 32000), (1, 1024), 0), alpha=1, beta=1, out=buf973)
        del primals_363
        buf976 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf973, buf976, 512, 32000, grid=grid(512), stream=stream0)
        buf979 = empty((), device='cuda', dtype=torch.float32)
        buf978 = empty((), device='cuda', dtype=torch.float32)
        buf1028 = buf979; del buf979  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_11.run(buf1028, primals_365, buf976, buf978, 1, 512, grid=grid(1), stream=stream0)
        return (buf1028, reinterpret_tensor(buf973, (1, 512, 32000), (16384000, 32000, 1), 0), primals_170, primals_176, primals_178, primals_184, primals_186, primals_192, primals_194, primals_200, primals_202, primals_208, primals_210, primals_216, primals_218, primals_224, primals_226, primals_232, primals_234, primals_240, primals_242, primals_248, primals_250, primals_256, primals_258, primals_264, primals_266, primals_272, primals_274, primals_280, primals_282, primals_288, primals_290, primals_296, primals_298, primals_304, primals_306, primals_312, primals_314, primals_320, primals_322, primals_328, primals_330, primals_336, primals_338, primals_344, primals_346, primals_352, primals_354, primals_360, primals_365, reinterpret_tensor(primals_364, (512, 1), (1, 512), 0), buf3, buf17, buf23, buf30, buf34, buf35, buf36, buf40, reinterpret_tensor(buf39, (512, 4096), (4096, 1), 0), buf44, buf48, buf63, buf70, buf74, buf75, buf76, buf80, reinterpret_tensor(buf79, (512, 4096), (4096, 1), 0), buf84, buf88, buf103, buf110, buf114, buf115, buf116, buf120, reinterpret_tensor(buf119, (512, 4096), (4096, 1), 0), buf124, buf128, buf143, buf150, buf154, buf155, buf156, buf160, reinterpret_tensor(buf159, (512, 4096), (4096, 1), 0), buf164, buf168, buf183, buf190, buf194, buf195, buf196, buf200, reinterpret_tensor(buf199, (512, 4096), (4096, 1), 0), buf204, buf208, buf223, buf230, buf234, buf235, buf236, buf240, reinterpret_tensor(buf239, (512, 4096), (4096, 1), 0), buf244, buf248, buf263, buf270, buf274, buf275, buf276, buf280, reinterpret_tensor(buf279, (512, 4096), (4096, 1), 0), buf284, buf288, buf303, buf310, buf314, buf315, buf316, buf320, reinterpret_tensor(buf319, (512, 4096), (4096, 1), 0), buf324, buf328, buf343, buf350, buf354, buf355, buf356, buf360, reinterpret_tensor(buf359, (512, 4096), (4096, 1), 0), buf364, buf368, buf383, buf390, buf394, buf395, buf396, buf400, reinterpret_tensor(buf399, (512, 4096), (4096, 1), 0), buf404, buf408, buf423, buf430, buf434, buf435, buf436, buf440, reinterpret_tensor(buf439, (512, 4096), (4096, 1), 0), buf444, buf448, buf463, buf470, buf474, buf475, buf476, buf480, reinterpret_tensor(buf479, (512, 4096), (4096, 1), 0), buf484, buf488, buf503, buf510, buf514, buf515, buf516, buf520, reinterpret_tensor(buf519, (512, 4096), (4096, 1), 0), buf524, buf528, buf543, buf550, buf554, buf555, buf556, buf560, reinterpret_tensor(buf559, (512, 4096), (4096, 1), 0), buf564, buf568, buf583, buf590, buf594, buf595, buf596, buf600, reinterpret_tensor(buf599, (512, 4096), (4096, 1), 0), buf604, buf608, buf623, buf630, buf634, buf635, buf636, buf640, reinterpret_tensor(buf639, (512, 4096), (4096, 1), 0), buf644, buf648, buf663, buf670, buf674, buf675, buf676, buf680, reinterpret_tensor(buf679, (512, 4096), (4096, 1), 0), buf684, buf688, buf703, buf710, buf714, buf715, buf716, buf720, reinterpret_tensor(buf719, (512, 4096), (4096, 1), 0), buf724, buf728, buf743, buf750, buf754, buf755, buf756, buf760, reinterpret_tensor(buf759, (512, 4096), (4096, 1), 0), buf764, buf768, buf783, buf790, buf794, buf795, buf796, buf800, reinterpret_tensor(buf799, (512, 4096), (4096, 1), 0), buf804, buf808, buf823, buf830, buf834, buf835, buf836, buf840, reinterpret_tensor(buf839, (512, 4096), (4096, 1), 0), buf844, buf848, buf863, buf870, buf874, buf875, buf876, buf880, reinterpret_tensor(buf879, (512, 4096), (4096, 1), 0), buf884, buf888, buf903, buf910, buf914, buf915, buf916, buf920, reinterpret_tensor(buf919, (512, 4096), (4096, 1), 0), buf924, buf928, buf943, buf950, buf954, buf955, buf956, buf960, reinterpret_tensor(buf959, (512, 4096), (4096, 1), 0), buf964, buf968, buf972, reinterpret_tensor(buf971, (512, 1024), (1024, 1), 0), buf976, buf978, reinterpret_tensor(primals_362, (32000, 1024), (1024, 1), 0), buf980, reinterpret_tensor(primals_358, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_356, (4096, 1024), (1024, 1), 0), buf981, reinterpret_tensor(buf945, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf946, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf942, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf932, (16, 64, 512), (64, 1, 1024), 0), buf940, reinterpret_tensor(buf936, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf933, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf934, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf931, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf929, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_164, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_163, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_162, (1, 1024, 1024), (1048576, 1, 1024), 0), buf982, reinterpret_tensor(primals_350, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_348, (4096, 1024), (1024, 1), 0), buf983, reinterpret_tensor(buf905, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf906, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf902, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf892, (16, 64, 512), (64, 1, 1024), 0), buf900, reinterpret_tensor(buf896, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf893, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf894, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf891, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf889, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_157, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_156, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_155, (1, 1024, 1024), (1048576, 1, 1024), 0), buf984, reinterpret_tensor(primals_342, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_340, (4096, 1024), (1024, 1), 0), buf985, reinterpret_tensor(buf865, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf866, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf862, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf852, (16, 64, 512), (64, 1, 1024), 0), buf860, reinterpret_tensor(buf856, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf853, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf854, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf851, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf849, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_150, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_149, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_148, (1, 1024, 1024), (1048576, 1, 1024), 0), buf986, reinterpret_tensor(primals_334, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_332, (4096, 1024), (1024, 1), 0), buf987, reinterpret_tensor(buf825, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf826, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf822, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf812, (16, 64, 512), (64, 1, 1024), 0), buf820, reinterpret_tensor(buf816, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf813, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf814, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf811, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf809, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_143, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_142, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_141, (1, 1024, 1024), (1048576, 1, 1024), 0), buf988, reinterpret_tensor(primals_326, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_324, (4096, 1024), (1024, 1), 0), buf989, reinterpret_tensor(buf785, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf786, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf782, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf772, (16, 64, 512), (64, 1, 1024), 0), buf780, reinterpret_tensor(buf776, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf773, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf774, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf771, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf769, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_136, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_135, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_134, (1, 1024, 1024), (1048576, 1, 1024), 0), buf990, reinterpret_tensor(primals_318, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_316, (4096, 1024), (1024, 1), 0), buf991, reinterpret_tensor(buf745, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf746, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf742, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf732, (16, 64, 512), (64, 1, 1024), 0), buf740, reinterpret_tensor(buf736, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf733, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf734, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf731, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf729, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_129, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_128, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_127, (1, 1024, 1024), (1048576, 1, 1024), 0), buf992, reinterpret_tensor(primals_310, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_308, (4096, 1024), (1024, 1), 0), buf993, reinterpret_tensor(buf705, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf706, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf702, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf692, (16, 64, 512), (64, 1, 1024), 0), buf700, reinterpret_tensor(buf696, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf693, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf694, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf691, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf689, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_122, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_121, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_120, (1, 1024, 1024), (1048576, 1, 1024), 0), buf994, reinterpret_tensor(primals_302, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_300, (4096, 1024), (1024, 1), 0), buf995, reinterpret_tensor(buf665, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf666, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf662, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf652, (16, 64, 512), (64, 1, 1024), 0), buf660, reinterpret_tensor(buf656, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf653, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf654, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf651, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf649, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_115, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_114, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_113, (1, 1024, 1024), (1048576, 1, 1024), 0), buf996, reinterpret_tensor(primals_294, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_292, (4096, 1024), (1024, 1), 0), buf997, reinterpret_tensor(buf625, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf626, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf622, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf612, (16, 64, 512), (64, 1, 1024), 0), buf620, reinterpret_tensor(buf616, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf613, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf614, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf611, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf609, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_108, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_107, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_106, (1, 1024, 1024), (1048576, 1, 1024), 0), buf998, reinterpret_tensor(primals_286, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_284, (4096, 1024), (1024, 1), 0), buf999, reinterpret_tensor(buf585, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf586, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf582, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf572, (16, 64, 512), (64, 1, 1024), 0), buf580, reinterpret_tensor(buf576, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf573, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf574, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf571, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf569, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_101, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_100, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_99, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1000, reinterpret_tensor(primals_278, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_276, (4096, 1024), (1024, 1), 0), buf1001, reinterpret_tensor(buf545, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf546, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf542, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf532, (16, 64, 512), (64, 1, 1024), 0), buf540, reinterpret_tensor(buf536, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf533, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf534, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf531, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf529, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_94, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_93, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_92, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1002, reinterpret_tensor(primals_270, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_268, (4096, 1024), (1024, 1), 0), buf1003, reinterpret_tensor(buf505, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf506, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf502, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf492, (16, 64, 512), (64, 1, 1024), 0), buf500, reinterpret_tensor(buf496, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf493, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf494, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf491, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf489, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_87, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_86, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_85, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1004, reinterpret_tensor(primals_262, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_260, (4096, 1024), (1024, 1), 0), buf1005, reinterpret_tensor(buf465, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf466, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf462, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf452, (16, 64, 512), (64, 1, 1024), 0), buf460, reinterpret_tensor(buf456, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf453, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf454, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf451, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf449, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_80, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_79, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_78, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1006, reinterpret_tensor(primals_254, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_252, (4096, 1024), (1024, 1), 0), buf1007, reinterpret_tensor(buf425, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf426, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf422, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf412, (16, 64, 512), (64, 1, 1024), 0), buf420, reinterpret_tensor(buf416, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf413, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf414, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf411, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf409, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_73, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_72, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_71, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1008, reinterpret_tensor(primals_246, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_244, (4096, 1024), (1024, 1), 0), buf1009, reinterpret_tensor(buf385, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf386, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf382, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf372, (16, 64, 512), (64, 1, 1024), 0), buf380, reinterpret_tensor(buf376, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf373, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf374, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf371, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf369, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_66, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_65, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_64, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1010, reinterpret_tensor(primals_238, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_236, (4096, 1024), (1024, 1), 0), buf1011, reinterpret_tensor(buf345, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf346, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf342, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf332, (16, 64, 512), (64, 1, 1024), 0), buf340, reinterpret_tensor(buf336, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf333, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf334, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf331, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf329, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_59, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_58, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_57, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1012, reinterpret_tensor(primals_230, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_228, (4096, 1024), (1024, 1), 0), buf1013, reinterpret_tensor(buf305, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf306, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf302, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf292, (16, 64, 512), (64, 1, 1024), 0), buf300, reinterpret_tensor(buf296, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf293, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf294, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf291, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf289, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_52, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_51, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_50, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1014, reinterpret_tensor(primals_222, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_220, (4096, 1024), (1024, 1), 0), buf1015, reinterpret_tensor(buf265, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf266, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf262, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf252, (16, 64, 512), (64, 1, 1024), 0), buf260, reinterpret_tensor(buf256, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf253, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf254, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf251, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf249, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_45, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_44, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_43, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1016, reinterpret_tensor(primals_214, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_212, (4096, 1024), (1024, 1), 0), buf1017, reinterpret_tensor(buf225, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf226, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf222, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf212, (16, 64, 512), (64, 1, 1024), 0), buf220, reinterpret_tensor(buf216, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf213, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf214, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf211, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf209, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_38, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_37, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_36, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1018, reinterpret_tensor(primals_206, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_204, (4096, 1024), (1024, 1), 0), buf1019, reinterpret_tensor(buf185, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf186, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf182, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf172, (16, 64, 512), (64, 1, 1024), 0), buf180, reinterpret_tensor(buf176, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf173, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf174, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf171, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf169, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_31, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_30, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_29, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1020, reinterpret_tensor(primals_198, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_196, (4096, 1024), (1024, 1), 0), buf1021, reinterpret_tensor(buf145, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf146, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf142, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf132, (16, 64, 512), (64, 1, 1024), 0), buf140, reinterpret_tensor(buf136, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf133, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf134, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf131, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf129, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_24, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_23, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_22, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1022, reinterpret_tensor(primals_190, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_188, (4096, 1024), (1024, 1), 0), buf1023, reinterpret_tensor(buf105, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf106, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf102, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf92, (16, 64, 512), (64, 1, 1024), 0), buf100, reinterpret_tensor(buf96, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf93, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf94, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf91, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf89, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_17, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_16, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_15, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1024, reinterpret_tensor(primals_182, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_180, (4096, 1024), (1024, 1), 0), buf1025, reinterpret_tensor(buf65, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf66, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf62, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf52, (16, 64, 512), (64, 1, 1024), 0), buf60, reinterpret_tensor(buf56, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf53, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf54, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf51, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf49, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(primals_10, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_9, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_8, (1, 1024, 1024), (1048576, 1, 1024), 0), buf1026, reinterpret_tensor(primals_174, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_172, (4096, 1024), (1024, 1), 0), buf1027, reinterpret_tensor(buf25, (1, 1024, 512), (0, 1, 1024), 0), reinterpret_tensor(buf26, (1, 1024, 1024), (0, 1, 1024), 0), reinterpret_tensor(buf22, (16, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf11, (16, 64, 512), (64, 1, 1024), 0), buf20, reinterpret_tensor(buf15, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf12, (16, 1024, 64), (64, 1024, 1), 0), reinterpret_tensor(buf13, (16, 64, 512), (64, 1, 1024), 0), reinterpret_tensor(buf10, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf2, (1, 1024, 512), (524288, 1, 1024), 0), reinterpret_tensor(primals_3, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_2, (1, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(primals_1, (1, 1024, 1024), (1048576, 1, 1024), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
