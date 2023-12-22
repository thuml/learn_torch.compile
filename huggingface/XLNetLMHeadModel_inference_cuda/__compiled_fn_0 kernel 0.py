
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


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwwd3wgdxzmdudeazga7zp3xkns2cf7n675i6cpp7wptsjzivaw.py
# Source Nodes: [add, add_1], Original ATen: [aten.add]
# add => add_2
# add_1 => add_3
triton_poi_fused_add_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': []},
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


cpp_fused_cat_2 = async_compile.cpp('''
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


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr57rovwextpp7pybjpgathshntfkpviz4ail4b4t7xkrbn3stv.py
# Source Nodes: [add_2, add_3, attn_prob, attn_score, bd_1], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
# add_2 => add_4
# add_3 => add_5
# attn_prob => amax, div_1, exp, sub, sum_1
# attn_score => mul_4
# bd_1 => index
triton_red_fused__softmax_add_index_select_mul_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7k/c7keci4jv36dstcy7mfhhxs4epuknysqbwj7yvywimu6riuwpog5.py
# Source Nodes: [attn_out], Original ATen: [aten.clone]
# attn_out => clone_3
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/od/codj5of734bh66u62tjrjplcl4iaynhe6yjg2tvbxvjhfvc2c3h3.py
# Source Nodes: [attn_out], Original ATen: [aten.clone]
# attn_out => clone_4
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4ope7ikvb3uib237d4c2juaqpj5hunz5y6znkujrryasroslnz.py
# Source Nodes: [attn_out_2, output_1], Original ATen: [aten.add, aten.native_layer_norm]
# attn_out_2 => add_6
# output_1 => add_7, add_8, mul_5, mul_6, rsqrt, sub_1, var_mean
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73bgg4xjymknge3asswlmg4crmhd2hhhduli7bitkzhf2md3yk7.py
# Source Nodes: [output_3], Original ATen: [aten.gelu]
# output_3 => add_9, erf, mul_7, mul_8, mul_9
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zy2atd6qerxr4oxcmfdvsy7l6gauhuafkgas6yk4tuzxv5cued.py
# Source Nodes: [add_5, cat_2], Original ATen: [aten.add, aten.native_layer_norm]
# add_5 => add_10
# cat_2 => add_11, add_12, mul_10, mul_11, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctev3z5usqji4bf6akwqgewyltojwjmxhqx5cnt4sfxyx5qwyqmb.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24, exp_24, sub_72, sum_25
triton_red_fused__log_softmax_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4ivedsqmpgmgopezjn6mplmh4fctg7znodxjuufsdymrsravr6q.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_5, div_25, full_default_1, ne_1, ne_2, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
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
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 32000
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 32000), "index out of bounds: 0 <= tmp7 < 32000")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg1_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg2_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg3_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg4_1, (16, 64), (64, 1))
    assert_size_stride(arg5_1, (16, 64), (64, 1))
    assert_size_stride(arg6_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg7_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg8_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg9_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg10_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg11_1, (16, 64), (64, 1))
    assert_size_stride(arg12_1, (16, 64), (64, 1))
    assert_size_stride(arg13_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg14_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg15_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg16_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg17_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg18_1, (16, 64), (64, 1))
    assert_size_stride(arg19_1, (16, 64), (64, 1))
    assert_size_stride(arg20_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg21_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg22_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg23_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg24_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg25_1, (16, 64), (64, 1))
    assert_size_stride(arg26_1, (16, 64), (64, 1))
    assert_size_stride(arg27_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg28_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg29_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg30_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg31_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg32_1, (16, 64), (64, 1))
    assert_size_stride(arg33_1, (16, 64), (64, 1))
    assert_size_stride(arg34_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg35_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg36_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg37_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg38_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg39_1, (16, 64), (64, 1))
    assert_size_stride(arg40_1, (16, 64), (64, 1))
    assert_size_stride(arg41_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg42_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg43_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg44_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg45_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg46_1, (16, 64), (64, 1))
    assert_size_stride(arg47_1, (16, 64), (64, 1))
    assert_size_stride(arg48_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg49_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg50_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg51_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg52_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg53_1, (16, 64), (64, 1))
    assert_size_stride(arg54_1, (16, 64), (64, 1))
    assert_size_stride(arg55_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg56_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg57_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg58_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg59_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg60_1, (16, 64), (64, 1))
    assert_size_stride(arg61_1, (16, 64), (64, 1))
    assert_size_stride(arg62_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg63_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg64_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg65_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg66_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg67_1, (16, 64), (64, 1))
    assert_size_stride(arg68_1, (16, 64), (64, 1))
    assert_size_stride(arg69_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg70_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg71_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg72_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg73_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg74_1, (16, 64), (64, 1))
    assert_size_stride(arg75_1, (16, 64), (64, 1))
    assert_size_stride(arg76_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg77_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg78_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg79_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg80_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg81_1, (16, 64), (64, 1))
    assert_size_stride(arg82_1, (16, 64), (64, 1))
    assert_size_stride(arg83_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg84_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg85_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg86_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg87_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg88_1, (16, 64), (64, 1))
    assert_size_stride(arg89_1, (16, 64), (64, 1))
    assert_size_stride(arg90_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg91_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg92_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg93_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg94_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg95_1, (16, 64), (64, 1))
    assert_size_stride(arg96_1, (16, 64), (64, 1))
    assert_size_stride(arg97_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg98_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg99_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg100_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg101_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg102_1, (16, 64), (64, 1))
    assert_size_stride(arg103_1, (16, 64), (64, 1))
    assert_size_stride(arg104_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg105_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg106_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg107_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg108_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg109_1, (16, 64), (64, 1))
    assert_size_stride(arg110_1, (16, 64), (64, 1))
    assert_size_stride(arg111_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg112_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg113_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg114_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg115_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg116_1, (16, 64), (64, 1))
    assert_size_stride(arg117_1, (16, 64), (64, 1))
    assert_size_stride(arg118_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg119_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg120_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg121_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg122_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg123_1, (16, 64), (64, 1))
    assert_size_stride(arg124_1, (16, 64), (64, 1))
    assert_size_stride(arg125_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg126_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg127_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg128_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg129_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg130_1, (16, 64), (64, 1))
    assert_size_stride(arg131_1, (16, 64), (64, 1))
    assert_size_stride(arg132_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg133_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg134_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg135_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg136_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg137_1, (16, 64), (64, 1))
    assert_size_stride(arg138_1, (16, 64), (64, 1))
    assert_size_stride(arg139_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg140_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg141_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg142_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg143_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg144_1, (16, 64), (64, 1))
    assert_size_stride(arg145_1, (16, 64), (64, 1))
    assert_size_stride(arg146_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg147_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg148_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg149_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg150_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg151_1, (16, 64), (64, 1))
    assert_size_stride(arg152_1, (16, 64), (64, 1))
    assert_size_stride(arg153_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg154_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg155_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg156_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg157_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg158_1, (16, 64), (64, 1))
    assert_size_stride(arg159_1, (16, 64), (64, 1))
    assert_size_stride(arg160_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg161_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg162_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg163_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg164_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg165_1, (16, 64), (64, 1))
    assert_size_stride(arg166_1, (16, 64), (64, 1))
    assert_size_stride(arg167_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg168_1, (32000, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg172_1, (4096, ), (1, ))
    assert_size_stride(arg173_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg180_1, (4096, ), (1, ))
    assert_size_stride(arg181_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg188_1, (4096, ), (1, ))
    assert_size_stride(arg189_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg196_1, (4096, ), (1, ))
    assert_size_stride(arg197_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg204_1, (4096, ), (1, ))
    assert_size_stride(arg205_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg212_1, (4096, ), (1, ))
    assert_size_stride(arg213_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg220_1, (4096, ), (1, ))
    assert_size_stride(arg221_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (1024, ), (1, ))
    assert_size_stride(arg224_1, (1024, ), (1, ))
    assert_size_stride(arg225_1, (1024, ), (1, ))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg228_1, (4096, ), (1, ))
    assert_size_stride(arg229_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, ), (1, ))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, ), (1, ))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg236_1, (4096, ), (1, ))
    assert_size_stride(arg237_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg244_1, (4096, ), (1, ))
    assert_size_stride(arg245_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, ), (1, ))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg252_1, (4096, ), (1, ))
    assert_size_stride(arg253_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (1024, ), (1, ))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg260_1, (4096, ), (1, ))
    assert_size_stride(arg261_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg268_1, (4096, ), (1, ))
    assert_size_stride(arg269_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg276_1, (4096, ), (1, ))
    assert_size_stride(arg277_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg284_1, (4096, ), (1, ))
    assert_size_stride(arg285_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg292_1, (4096, ), (1, ))
    assert_size_stride(arg293_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg300_1, (4096, ), (1, ))
    assert_size_stride(arg301_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg308_1, (4096, ), (1, ))
    assert_size_stride(arg309_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg316_1, (4096, ), (1, ))
    assert_size_stride(arg317_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (1024, ), (1, ))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg324_1, (4096, ), (1, ))
    assert_size_stride(arg325_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg332_1, (4096, ), (1, ))
    assert_size_stride(arg333_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (1024, ), (1, ))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg340_1, (4096, ), (1, ))
    assert_size_stride(arg341_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg348_1, (4096, ), (1, ))
    assert_size_stride(arg349_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (1024, ), (1, ))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg356_1, (4096, ), (1, ))
    assert_size_stride(arg357_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (32000, 1024), (1024, 1))
    assert_size_stride(arg362_1, (32000, ), (1, ))
    assert_size_stride(arg363_1, (1, 512), (512, 1))
    assert_size_stride(arg364_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [word_emb_k], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(arg363_1, arg168_1, buf0, 524288, grid=grid(524288), stream=stream0)
        del arg168_1
        del arg363_1
        buf1 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [q_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(arg0_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf1)
        del arg0_1
        buf2 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg1_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf2)
        del arg1_1
        buf3 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        buf8 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf1, arg4_1, arg5_1, buf3, buf8, 524288, grid=grid(524288), stream=stream0)
        del arg4_1
        del arg5_1
        buf4 = empty((16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [ac], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf2, (16, 64, 512), (64, 1, 1024), 0), out=buf4)
    buf5 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_cat_2(c_void_p(buf5.data_ptr()))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf6 = empty((1024, 1, 1024), device='cuda', dtype=torch.float32)
        buf6.copy_(reinterpret_tensor(buf5, (1024, 1, 1024), (1024, 0, 1), 0))
        del buf5
        buf7 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_head_r], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg3_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf7)
        del arg3_1
        buf9 = empty((16, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [bd], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf7, (16, 64, 1024), (64, 1, 1024), 0), out=buf9)
        buf13 = empty((1, 16, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, attn_prob, attn_score, bd_1], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf4, buf9, buf13, 8192, 512, grid=grid(8192), stream=stream0)
        buf12 = reinterpret_tensor(buf8, (1, 512, 1024), (524288, 1024, 1), 0); del buf8  # reuse
        # Source Nodes: [v_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg2_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf12)
        del arg2_1
        buf14 = reinterpret_tensor(buf3, (16, 512, 64), (32768, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [attn_vec], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf12, (16, 512, 64), (64, 1024, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf12, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf12  # reuse
        # Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf16 = reinterpret_tensor(buf7, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf7  # reuse
        # Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg6_1, buf16, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg6_1
        buf17 = reinterpret_tensor(buf14, (1, 512, 1024), (524288, 1024, 1), 0); del buf14  # reuse
        # Source Nodes: [attn_out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf16, (1, 1024, 1024), (0, 1024, 1), 0), out=buf17)
        buf21 = reinterpret_tensor(buf15, (512, 1, 1024), (1024, 1024, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_out_2, output_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf17, buf0, arg169_1, arg170_1, buf21, 512, 1024, grid=grid(512), stream=stream0)
        del arg169_1
        del arg170_1
        buf22 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 4096), (1, 1024), 0), out=buf22)
        del arg171_1
        buf23 = reinterpret_tensor(buf22, (512, 1, 4096), (4096, 4096, 1), 0); del buf22  # reuse
        # Source Nodes: [output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf23, arg172_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg172_1
        buf24 = reinterpret_tensor(buf17, (512, 1024), (1024, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg173_1, (4096, 1024), (1, 4096), 0), out=buf24)
        del arg173_1
        buf28 = reinterpret_tensor(buf2, (512, 1, 1024), (1024, 1024, 1), 0); del buf2  # reuse
        # Source Nodes: [add_5, cat_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf24, arg174_1, buf21, arg175_1, arg176_1, buf28, 512, 1024, grid=grid(512), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        buf29 = reinterpret_tensor(buf24, (1, 512, 1024), (524288, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [q_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg7_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf29)
        del arg7_1
        buf30 = reinterpret_tensor(buf21, (1, 512, 1024), (524288, 1024, 1), 0); del buf21  # reuse
        # Source Nodes: [k_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg8_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf30)
        del arg8_1
        buf31 = reinterpret_tensor(buf1, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf1  # reuse
        buf34 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf29, arg11_1, arg12_1, buf31, buf34, 524288, grid=grid(524288), stream=stream0)
        del arg11_1
        del arg12_1
        buf32 = reinterpret_tensor(buf13, (16, 512, 512), (262144, 512, 1), 0); del buf13  # reuse
        # Source Nodes: [ac_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf30, (16, 64, 512), (64, 1, 1024), 0), out=buf32)
        buf33 = reinterpret_tensor(buf16, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf16  # reuse
        # Source Nodes: [k_head_r_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg10_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf33)
        del arg10_1
        buf35 = buf9; del buf9  # reuse
        # Source Nodes: [bd_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf33, (16, 64, 1024), (64, 1, 1024), 0), out=buf35)
        buf39 = reinterpret_tensor(buf4, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf4  # reuse
        # Source Nodes: [add_8, add_9, attn_prob_2, attn_score_1, bd_3], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf32, buf35, buf39, 8192, 512, grid=grid(8192), stream=stream0)
        buf38 = reinterpret_tensor(buf34, (1, 512, 1024), (524288, 1024, 1), 0); del buf34  # reuse
        # Source Nodes: [v_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg9_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf38)
        del arg9_1
        buf40 = reinterpret_tensor(buf31, (16, 512, 64), (32768, 64, 1), 0); del buf31  # reuse
        # Source Nodes: [attn_vec_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf38, (16, 512, 64), (64, 1024, 1), 0), out=buf40)
        buf41 = reinterpret_tensor(buf38, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf38  # reuse
        # Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf40, buf41, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf42 = reinterpret_tensor(buf33, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf33  # reuse
        # Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg13_1, buf42, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg13_1
        buf43 = reinterpret_tensor(buf40, (1, 512, 1024), (524288, 1024, 1), 0); del buf40  # reuse
        # Source Nodes: [attn_out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf42, (1, 1024, 1024), (0, 1024, 1), 0), out=buf43)
        buf47 = reinterpret_tensor(buf41, (512, 1, 1024), (1024, 1024, 1), 0); del buf41  # reuse
        # Source Nodes: [attn_out_5, output_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf43, buf28, arg177_1, arg178_1, buf47, 512, 1024, grid=grid(512), stream=stream0)
        del arg177_1
        del arg178_1
        buf48 = reinterpret_tensor(buf23, (512, 4096), (4096, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg179_1, (1024, 4096), (1, 1024), 0), out=buf48)
        del arg179_1
        buf49 = reinterpret_tensor(buf48, (512, 1, 4096), (4096, 4096, 1), 0); del buf48  # reuse
        # Source Nodes: [output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf49, arg180_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg180_1
        buf50 = reinterpret_tensor(buf43, (512, 1024), (1024, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg181_1, (4096, 1024), (1, 4096), 0), out=buf50)
        del arg181_1
        buf54 = reinterpret_tensor(buf30, (512, 1, 1024), (1024, 1024, 1), 0); del buf30  # reuse
        # Source Nodes: [add_11, cat_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf50, arg182_1, buf47, arg183_1, arg184_1, buf54, 512, 1024, grid=grid(512), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        buf55 = reinterpret_tensor(buf50, (1, 512, 1024), (524288, 1024, 1), 0); del buf50  # reuse
        # Source Nodes: [q_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg14_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf55)
        del arg14_1
        buf56 = reinterpret_tensor(buf47, (1, 512, 1024), (524288, 1024, 1), 0); del buf47  # reuse
        # Source Nodes: [k_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg15_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf56)
        del arg15_1
        buf57 = reinterpret_tensor(buf29, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf29  # reuse
        buf60 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, add_13], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf55, arg18_1, arg19_1, buf57, buf60, 524288, grid=grid(524288), stream=stream0)
        del arg18_1
        del arg19_1
        buf58 = reinterpret_tensor(buf39, (16, 512, 512), (262144, 512, 1), 0); del buf39  # reuse
        # Source Nodes: [ac_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf56, (16, 64, 512), (64, 1, 1024), 0), out=buf58)
        buf59 = reinterpret_tensor(buf42, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf42  # reuse
        # Source Nodes: [k_head_r_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg17_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf59)
        del arg17_1
        buf61 = buf35; del buf35  # reuse
        # Source Nodes: [bd_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf59, (16, 64, 1024), (64, 1, 1024), 0), out=buf61)
        buf65 = reinterpret_tensor(buf32, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf32  # reuse
        # Source Nodes: [add_14, add_15, attn_prob_4, attn_score_2, bd_5], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf58, buf61, buf65, 8192, 512, grid=grid(8192), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (1, 512, 1024), (524288, 1024, 1), 0); del buf60  # reuse
        # Source Nodes: [v_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg16_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf64)
        del arg16_1
        buf66 = reinterpret_tensor(buf57, (16, 512, 64), (32768, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [attn_vec_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf64, (16, 512, 64), (64, 1024, 1), 0), out=buf66)
        buf67 = reinterpret_tensor(buf64, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf64  # reuse
        # Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf66, buf67, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf68 = reinterpret_tensor(buf59, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf59  # reuse
        # Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg20_1, buf68, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg20_1
        buf69 = reinterpret_tensor(buf66, (1, 512, 1024), (524288, 1024, 1), 0); del buf66  # reuse
        # Source Nodes: [attn_out_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf68, (1, 1024, 1024), (0, 1024, 1), 0), out=buf69)
        buf73 = reinterpret_tensor(buf67, (512, 1, 1024), (1024, 1024, 1), 0); del buf67  # reuse
        # Source Nodes: [attn_out_8, output_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf69, buf54, arg185_1, arg186_1, buf73, 512, 1024, grid=grid(512), stream=stream0)
        del arg185_1
        del arg186_1
        buf74 = reinterpret_tensor(buf49, (512, 4096), (4096, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 4096), (1, 1024), 0), out=buf74)
        del arg187_1
        buf75 = reinterpret_tensor(buf74, (512, 1, 4096), (4096, 4096, 1), 0); del buf74  # reuse
        # Source Nodes: [output_19], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf75, arg188_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg188_1
        buf76 = reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg189_1, (4096, 1024), (1, 4096), 0), out=buf76)
        del arg189_1
        buf80 = reinterpret_tensor(buf56, (512, 1, 1024), (1024, 1024, 1), 0); del buf56  # reuse
        # Source Nodes: [add_17, cat_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf76, arg190_1, buf73, arg191_1, arg192_1, buf80, 512, 1024, grid=grid(512), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        buf81 = reinterpret_tensor(buf76, (1, 512, 1024), (524288, 1024, 1), 0); del buf76  # reuse
        # Source Nodes: [q_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg21_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf81)
        del arg21_1
        buf82 = reinterpret_tensor(buf73, (1, 512, 1024), (524288, 1024, 1), 0); del buf73  # reuse
        # Source Nodes: [k_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg22_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf82)
        del arg22_1
        buf83 = reinterpret_tensor(buf55, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf55  # reuse
        buf86 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf81, arg25_1, arg26_1, buf83, buf86, 524288, grid=grid(524288), stream=stream0)
        del arg25_1
        del arg26_1
        buf84 = reinterpret_tensor(buf65, (16, 512, 512), (262144, 512, 1), 0); del buf65  # reuse
        # Source Nodes: [ac_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf82, (16, 64, 512), (64, 1, 1024), 0), out=buf84)
        buf85 = reinterpret_tensor(buf68, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf68  # reuse
        # Source Nodes: [k_head_r_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg24_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf85)
        del arg24_1
        buf87 = buf61; del buf61  # reuse
        # Source Nodes: [bd_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf85, (16, 64, 1024), (64, 1, 1024), 0), out=buf87)
        buf91 = reinterpret_tensor(buf58, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf58  # reuse
        # Source Nodes: [add_20, add_21, attn_prob_6, attn_score_3, bd_7], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf84, buf87, buf91, 8192, 512, grid=grid(8192), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (1, 512, 1024), (524288, 1024, 1), 0); del buf86  # reuse
        # Source Nodes: [v_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg23_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf90)
        del arg23_1
        buf92 = reinterpret_tensor(buf83, (16, 512, 64), (32768, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_vec_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf90, (16, 512, 64), (64, 1024, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf90, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf90  # reuse
        # Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf92, buf93, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf94 = reinterpret_tensor(buf85, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf85  # reuse
        # Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg27_1, buf94, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg27_1
        buf95 = reinterpret_tensor(buf92, (1, 512, 1024), (524288, 1024, 1), 0); del buf92  # reuse
        # Source Nodes: [attn_out_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf94, (1, 1024, 1024), (0, 1024, 1), 0), out=buf95)
        buf99 = reinterpret_tensor(buf93, (512, 1, 1024), (1024, 1024, 1), 0); del buf93  # reuse
        # Source Nodes: [attn_out_11, output_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf95, buf80, arg193_1, arg194_1, buf99, 512, 1024, grid=grid(512), stream=stream0)
        del arg193_1
        del arg194_1
        buf100 = reinterpret_tensor(buf75, (512, 4096), (4096, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg195_1, (1024, 4096), (1, 1024), 0), out=buf100)
        del arg195_1
        buf101 = reinterpret_tensor(buf100, (512, 1, 4096), (4096, 4096, 1), 0); del buf100  # reuse
        # Source Nodes: [output_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf101, arg196_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg196_1
        buf102 = reinterpret_tensor(buf95, (512, 1024), (1024, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg197_1, (4096, 1024), (1, 4096), 0), out=buf102)
        del arg197_1
        buf106 = reinterpret_tensor(buf82, (512, 1, 1024), (1024, 1024, 1), 0); del buf82  # reuse
        # Source Nodes: [add_23, cat_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf102, arg198_1, buf99, arg199_1, arg200_1, buf106, 512, 1024, grid=grid(512), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        buf107 = reinterpret_tensor(buf99, (1, 512, 1024), (524288, 1024, 1), 0); del buf99  # reuse
        # Source Nodes: [q_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg28_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf107)
        del arg28_1
        buf108 = reinterpret_tensor(buf102, (1, 512, 1024), (524288, 1024, 1), 0); del buf102  # reuse
        # Source Nodes: [k_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg29_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf108)
        del arg29_1
        buf109 = reinterpret_tensor(buf81, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf81  # reuse
        buf112 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, add_25], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf107, arg32_1, arg33_1, buf109, buf112, 524288, grid=grid(524288), stream=stream0)
        del arg32_1
        del arg33_1
        buf110 = reinterpret_tensor(buf91, (16, 512, 512), (262144, 512, 1), 0); del buf91  # reuse
        # Source Nodes: [ac_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf108, (16, 64, 512), (64, 1, 1024), 0), out=buf110)
        buf111 = reinterpret_tensor(buf94, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
        # Source Nodes: [k_head_r_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg31_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf111)
        del arg31_1
        buf113 = buf87; del buf87  # reuse
        # Source Nodes: [bd_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf112, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf111, (16, 64, 1024), (64, 1, 1024), 0), out=buf113)
        buf117 = reinterpret_tensor(buf84, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf84  # reuse
        # Source Nodes: [add_26, add_27, attn_prob_8, attn_score_4, bd_9], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf110, buf113, buf117, 8192, 512, grid=grid(8192), stream=stream0)
        buf116 = reinterpret_tensor(buf112, (1, 512, 1024), (524288, 1024, 1), 0); del buf112  # reuse
        # Source Nodes: [v_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg30_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf116)
        del arg30_1
        buf118 = reinterpret_tensor(buf109, (16, 512, 64), (32768, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [attn_vec_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf117, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf116, (16, 512, 64), (64, 1024, 1), 0), out=buf118)
        buf119 = reinterpret_tensor(buf116, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf116  # reuse
        # Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf118, buf119, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf120 = reinterpret_tensor(buf111, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf111  # reuse
        # Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg34_1, buf120, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg34_1
        buf121 = reinterpret_tensor(buf118, (1, 512, 1024), (524288, 1024, 1), 0); del buf118  # reuse
        # Source Nodes: [attn_out_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf120, (1, 1024, 1024), (0, 1024, 1), 0), out=buf121)
        buf125 = reinterpret_tensor(buf119, (512, 1, 1024), (1024, 1024, 1), 0); del buf119  # reuse
        # Source Nodes: [attn_out_14, output_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf121, buf106, arg201_1, arg202_1, buf125, 512, 1024, grid=grid(512), stream=stream0)
        del arg201_1
        del arg202_1
        buf126 = reinterpret_tensor(buf101, (512, 4096), (4096, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 4096), (1, 1024), 0), out=buf126)
        del arg203_1
        buf127 = reinterpret_tensor(buf126, (512, 1, 4096), (4096, 4096, 1), 0); del buf126  # reuse
        # Source Nodes: [output_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf127, arg204_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg204_1
        buf128 = reinterpret_tensor(buf121, (512, 1024), (1024, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg205_1, (4096, 1024), (1, 4096), 0), out=buf128)
        del arg205_1
        buf132 = reinterpret_tensor(buf108, (512, 1, 1024), (1024, 1024, 1), 0); del buf108  # reuse
        # Source Nodes: [add_29, cat_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf128, arg206_1, buf125, arg207_1, arg208_1, buf132, 512, 1024, grid=grid(512), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        buf133 = reinterpret_tensor(buf128, (1, 512, 1024), (524288, 1024, 1), 0); del buf128  # reuse
        # Source Nodes: [q_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg35_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf133)
        del arg35_1
        buf134 = reinterpret_tensor(buf125, (1, 512, 1024), (524288, 1024, 1), 0); del buf125  # reuse
        # Source Nodes: [k_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg36_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf134)
        del arg36_1
        buf135 = reinterpret_tensor(buf107, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf107  # reuse
        buf138 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf133, arg39_1, arg40_1, buf135, buf138, 524288, grid=grid(524288), stream=stream0)
        del arg39_1
        del arg40_1
        buf136 = reinterpret_tensor(buf117, (16, 512, 512), (262144, 512, 1), 0); del buf117  # reuse
        # Source Nodes: [ac_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf134, (16, 64, 512), (64, 1, 1024), 0), out=buf136)
        buf137 = reinterpret_tensor(buf120, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf120  # reuse
        # Source Nodes: [k_head_r_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg38_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf137)
        del arg38_1
        buf139 = buf113; del buf113  # reuse
        # Source Nodes: [bd_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf138, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf137, (16, 64, 1024), (64, 1, 1024), 0), out=buf139)
        buf143 = reinterpret_tensor(buf110, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf110  # reuse
        # Source Nodes: [add_32, add_33, attn_prob_10, attn_score_5, bd_11], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf136, buf139, buf143, 8192, 512, grid=grid(8192), stream=stream0)
        buf142 = reinterpret_tensor(buf138, (1, 512, 1024), (524288, 1024, 1), 0); del buf138  # reuse
        # Source Nodes: [v_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg37_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf142)
        del arg37_1
        buf144 = reinterpret_tensor(buf135, (16, 512, 64), (32768, 64, 1), 0); del buf135  # reuse
        # Source Nodes: [attn_vec_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf142, (16, 512, 64), (64, 1024, 1), 0), out=buf144)
        buf145 = reinterpret_tensor(buf142, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf142  # reuse
        # Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf144, buf145, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf146 = reinterpret_tensor(buf137, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf137  # reuse
        # Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg41_1, buf146, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg41_1
        buf147 = reinterpret_tensor(buf144, (1, 512, 1024), (524288, 1024, 1), 0); del buf144  # reuse
        # Source Nodes: [attn_out_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf146, (1, 1024, 1024), (0, 1024, 1), 0), out=buf147)
        buf151 = reinterpret_tensor(buf145, (512, 1, 1024), (1024, 1024, 1), 0); del buf145  # reuse
        # Source Nodes: [attn_out_17, output_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf147, buf132, arg209_1, arg210_1, buf151, 512, 1024, grid=grid(512), stream=stream0)
        del arg209_1
        del arg210_1
        buf152 = reinterpret_tensor(buf127, (512, 4096), (4096, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg211_1, (1024, 4096), (1, 1024), 0), out=buf152)
        del arg211_1
        buf153 = reinterpret_tensor(buf152, (512, 1, 4096), (4096, 4096, 1), 0); del buf152  # reuse
        # Source Nodes: [output_43], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf153, arg212_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg212_1
        buf154 = reinterpret_tensor(buf147, (512, 1024), (1024, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg213_1, (4096, 1024), (1, 4096), 0), out=buf154)
        del arg213_1
        buf158 = reinterpret_tensor(buf134, (512, 1, 1024), (1024, 1024, 1), 0); del buf134  # reuse
        # Source Nodes: [add_35, cat_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf154, arg214_1, buf151, arg215_1, arg216_1, buf158, 512, 1024, grid=grid(512), stream=stream0)
        del arg214_1
        del arg215_1
        del arg216_1
        buf159 = reinterpret_tensor(buf154, (1, 512, 1024), (524288, 1024, 1), 0); del buf154  # reuse
        # Source Nodes: [q_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg42_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf159)
        del arg42_1
        buf160 = reinterpret_tensor(buf151, (1, 512, 1024), (524288, 1024, 1), 0); del buf151  # reuse
        # Source Nodes: [k_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg43_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf160)
        del arg43_1
        buf161 = reinterpret_tensor(buf133, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf133  # reuse
        buf164 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, add_37], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf159, arg46_1, arg47_1, buf161, buf164, 524288, grid=grid(524288), stream=stream0)
        del arg46_1
        del arg47_1
        buf162 = reinterpret_tensor(buf143, (16, 512, 512), (262144, 512, 1), 0); del buf143  # reuse
        # Source Nodes: [ac_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf160, (16, 64, 512), (64, 1, 1024), 0), out=buf162)
        buf163 = reinterpret_tensor(buf146, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf146  # reuse
        # Source Nodes: [k_head_r_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg45_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf163)
        del arg45_1
        buf165 = buf139; del buf139  # reuse
        # Source Nodes: [bd_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf163, (16, 64, 1024), (64, 1, 1024), 0), out=buf165)
        buf169 = reinterpret_tensor(buf136, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf136  # reuse
        # Source Nodes: [add_38, add_39, attn_prob_12, attn_score_6, bd_13], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf162, buf165, buf169, 8192, 512, grid=grid(8192), stream=stream0)
        buf168 = reinterpret_tensor(buf164, (1, 512, 1024), (524288, 1024, 1), 0); del buf164  # reuse
        # Source Nodes: [v_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg44_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf168)
        del arg44_1
        buf170 = reinterpret_tensor(buf161, (16, 512, 64), (32768, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [attn_vec_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf168, (16, 512, 64), (64, 1024, 1), 0), out=buf170)
        buf171 = reinterpret_tensor(buf168, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf168  # reuse
        # Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf170, buf171, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf172 = reinterpret_tensor(buf163, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf163  # reuse
        # Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg48_1, buf172, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg48_1
        buf173 = reinterpret_tensor(buf170, (1, 512, 1024), (524288, 1024, 1), 0); del buf170  # reuse
        # Source Nodes: [attn_out_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf172, (1, 1024, 1024), (0, 1024, 1), 0), out=buf173)
        buf177 = reinterpret_tensor(buf171, (512, 1, 1024), (1024, 1024, 1), 0); del buf171  # reuse
        # Source Nodes: [attn_out_20, output_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf173, buf158, arg217_1, arg218_1, buf177, 512, 1024, grid=grid(512), stream=stream0)
        del arg217_1
        del arg218_1
        buf178 = reinterpret_tensor(buf153, (512, 4096), (4096, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 4096), (1, 1024), 0), out=buf178)
        del arg219_1
        buf179 = reinterpret_tensor(buf178, (512, 1, 4096), (4096, 4096, 1), 0); del buf178  # reuse
        # Source Nodes: [output_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf179, arg220_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg220_1
        buf180 = reinterpret_tensor(buf173, (512, 1024), (1024, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg221_1, (4096, 1024), (1, 4096), 0), out=buf180)
        del arg221_1
        buf184 = reinterpret_tensor(buf160, (512, 1, 1024), (1024, 1024, 1), 0); del buf160  # reuse
        # Source Nodes: [add_41, cat_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf180, arg222_1, buf177, arg223_1, arg224_1, buf184, 512, 1024, grid=grid(512), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        buf185 = reinterpret_tensor(buf180, (1, 512, 1024), (524288, 1024, 1), 0); del buf180  # reuse
        # Source Nodes: [q_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg49_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf185)
        del arg49_1
        buf186 = reinterpret_tensor(buf177, (1, 512, 1024), (524288, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [k_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg50_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf186)
        del arg50_1
        buf187 = reinterpret_tensor(buf159, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf159  # reuse
        buf190 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf185, arg53_1, arg54_1, buf187, buf190, 524288, grid=grid(524288), stream=stream0)
        del arg53_1
        del arg54_1
        buf188 = reinterpret_tensor(buf169, (16, 512, 512), (262144, 512, 1), 0); del buf169  # reuse
        # Source Nodes: [ac_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf186, (16, 64, 512), (64, 1, 1024), 0), out=buf188)
        buf189 = reinterpret_tensor(buf172, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf172  # reuse
        # Source Nodes: [k_head_r_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg52_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf189)
        del arg52_1
        buf191 = buf165; del buf165  # reuse
        # Source Nodes: [bd_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf189, (16, 64, 1024), (64, 1, 1024), 0), out=buf191)
        buf195 = reinterpret_tensor(buf162, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf162  # reuse
        # Source Nodes: [add_44, add_45, attn_prob_14, attn_score_7, bd_15], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf188, buf191, buf195, 8192, 512, grid=grid(8192), stream=stream0)
        buf194 = reinterpret_tensor(buf190, (1, 512, 1024), (524288, 1024, 1), 0); del buf190  # reuse
        # Source Nodes: [v_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg51_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf194)
        del arg51_1
        buf196 = reinterpret_tensor(buf187, (16, 512, 64), (32768, 64, 1), 0); del buf187  # reuse
        # Source Nodes: [attn_vec_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf195, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf194, (16, 512, 64), (64, 1024, 1), 0), out=buf196)
        buf197 = reinterpret_tensor(buf194, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf194  # reuse
        # Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf196, buf197, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf198 = reinterpret_tensor(buf189, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf189  # reuse
        # Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg55_1, buf198, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg55_1
        buf199 = reinterpret_tensor(buf196, (1, 512, 1024), (524288, 1024, 1), 0); del buf196  # reuse
        # Source Nodes: [attn_out_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf197, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf198, (1, 1024, 1024), (0, 1024, 1), 0), out=buf199)
        buf203 = reinterpret_tensor(buf197, (512, 1, 1024), (1024, 1024, 1), 0); del buf197  # reuse
        # Source Nodes: [attn_out_23, output_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf199, buf184, arg225_1, arg226_1, buf203, 512, 1024, grid=grid(512), stream=stream0)
        del arg225_1
        del arg226_1
        buf204 = reinterpret_tensor(buf179, (512, 4096), (4096, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg227_1, (1024, 4096), (1, 1024), 0), out=buf204)
        del arg227_1
        buf205 = reinterpret_tensor(buf204, (512, 1, 4096), (4096, 4096, 1), 0); del buf204  # reuse
        # Source Nodes: [output_59], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf205, arg228_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg228_1
        buf206 = reinterpret_tensor(buf199, (512, 1024), (1024, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg229_1, (4096, 1024), (1, 4096), 0), out=buf206)
        del arg229_1
        buf210 = reinterpret_tensor(buf186, (512, 1, 1024), (1024, 1024, 1), 0); del buf186  # reuse
        # Source Nodes: [add_47, cat_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf206, arg230_1, buf203, arg231_1, arg232_1, buf210, 512, 1024, grid=grid(512), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        buf211 = reinterpret_tensor(buf206, (1, 512, 1024), (524288, 1024, 1), 0); del buf206  # reuse
        # Source Nodes: [q_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg56_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf211)
        del arg56_1
        buf212 = reinterpret_tensor(buf203, (1, 512, 1024), (524288, 1024, 1), 0); del buf203  # reuse
        # Source Nodes: [k_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg57_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf212)
        del arg57_1
        buf213 = reinterpret_tensor(buf185, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf185  # reuse
        buf216 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, add_49], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf211, arg60_1, arg61_1, buf213, buf216, 524288, grid=grid(524288), stream=stream0)
        del arg60_1
        del arg61_1
        buf214 = reinterpret_tensor(buf195, (16, 512, 512), (262144, 512, 1), 0); del buf195  # reuse
        # Source Nodes: [ac_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf212, (16, 64, 512), (64, 1, 1024), 0), out=buf214)
        buf215 = reinterpret_tensor(buf198, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf198  # reuse
        # Source Nodes: [k_head_r_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg59_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf215)
        del arg59_1
        buf217 = buf191; del buf191  # reuse
        # Source Nodes: [bd_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf215, (16, 64, 1024), (64, 1, 1024), 0), out=buf217)
        buf221 = reinterpret_tensor(buf188, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf188  # reuse
        # Source Nodes: [add_50, add_51, attn_prob_16, attn_score_8, bd_17], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf214, buf217, buf221, 8192, 512, grid=grid(8192), stream=stream0)
        buf220 = reinterpret_tensor(buf216, (1, 512, 1024), (524288, 1024, 1), 0); del buf216  # reuse
        # Source Nodes: [v_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg58_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf220)
        del arg58_1
        buf222 = reinterpret_tensor(buf213, (16, 512, 64), (32768, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [attn_vec_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf221, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf220, (16, 512, 64), (64, 1024, 1), 0), out=buf222)
        buf223 = reinterpret_tensor(buf220, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf220  # reuse
        # Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf222, buf223, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf224 = reinterpret_tensor(buf215, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf215  # reuse
        # Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg62_1, buf224, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg62_1
        buf225 = reinterpret_tensor(buf222, (1, 512, 1024), (524288, 1024, 1), 0); del buf222  # reuse
        # Source Nodes: [attn_out_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf224, (1, 1024, 1024), (0, 1024, 1), 0), out=buf225)
        buf229 = reinterpret_tensor(buf223, (512, 1, 1024), (1024, 1024, 1), 0); del buf223  # reuse
        # Source Nodes: [attn_out_26, output_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf225, buf210, arg233_1, arg234_1, buf229, 512, 1024, grid=grid(512), stream=stream0)
        del arg233_1
        del arg234_1
        buf230 = reinterpret_tensor(buf205, (512, 4096), (4096, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 4096), (1, 1024), 0), out=buf230)
        del arg235_1
        buf231 = reinterpret_tensor(buf230, (512, 1, 4096), (4096, 4096, 1), 0); del buf230  # reuse
        # Source Nodes: [output_67], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf231, arg236_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg236_1
        buf232 = reinterpret_tensor(buf225, (512, 1024), (1024, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg237_1, (4096, 1024), (1, 4096), 0), out=buf232)
        del arg237_1
        buf236 = reinterpret_tensor(buf212, (512, 1, 1024), (1024, 1024, 1), 0); del buf212  # reuse
        # Source Nodes: [add_53, cat_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf232, arg238_1, buf229, arg239_1, arg240_1, buf236, 512, 1024, grid=grid(512), stream=stream0)
        del arg238_1
        del arg239_1
        del arg240_1
        buf237 = reinterpret_tensor(buf232, (1, 512, 1024), (524288, 1024, 1), 0); del buf232  # reuse
        # Source Nodes: [q_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg63_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf237)
        del arg63_1
        buf238 = reinterpret_tensor(buf229, (1, 512, 1024), (524288, 1024, 1), 0); del buf229  # reuse
        # Source Nodes: [k_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg64_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf238)
        del arg64_1
        buf239 = reinterpret_tensor(buf211, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf211  # reuse
        buf242 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_54, add_55], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf237, arg67_1, arg68_1, buf239, buf242, 524288, grid=grid(524288), stream=stream0)
        del arg67_1
        del arg68_1
        buf240 = reinterpret_tensor(buf221, (16, 512, 512), (262144, 512, 1), 0); del buf221  # reuse
        # Source Nodes: [ac_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf238, (16, 64, 512), (64, 1, 1024), 0), out=buf240)
        buf241 = reinterpret_tensor(buf224, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf224  # reuse
        # Source Nodes: [k_head_r_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg66_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf241)
        del arg66_1
        buf243 = buf217; del buf217  # reuse
        # Source Nodes: [bd_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf241, (16, 64, 1024), (64, 1, 1024), 0), out=buf243)
        buf247 = reinterpret_tensor(buf214, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf214  # reuse
        # Source Nodes: [add_56, add_57, attn_prob_18, attn_score_9, bd_19], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf240, buf243, buf247, 8192, 512, grid=grid(8192), stream=stream0)
        buf246 = reinterpret_tensor(buf242, (1, 512, 1024), (524288, 1024, 1), 0); del buf242  # reuse
        # Source Nodes: [v_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg65_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf246)
        del arg65_1
        buf248 = reinterpret_tensor(buf239, (16, 512, 64), (32768, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [attn_vec_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf246, (16, 512, 64), (64, 1024, 1), 0), out=buf248)
        buf249 = reinterpret_tensor(buf246, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf246  # reuse
        # Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf248, buf249, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf250 = reinterpret_tensor(buf241, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf241  # reuse
        # Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg69_1, buf250, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg69_1
        buf251 = reinterpret_tensor(buf248, (1, 512, 1024), (524288, 1024, 1), 0); del buf248  # reuse
        # Source Nodes: [attn_out_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf250, (1, 1024, 1024), (0, 1024, 1), 0), out=buf251)
        buf255 = reinterpret_tensor(buf249, (512, 1, 1024), (1024, 1024, 1), 0); del buf249  # reuse
        # Source Nodes: [attn_out_29, output_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf251, buf236, arg241_1, arg242_1, buf255, 512, 1024, grid=grid(512), stream=stream0)
        del arg241_1
        del arg242_1
        buf256 = reinterpret_tensor(buf231, (512, 4096), (4096, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg243_1, (1024, 4096), (1, 1024), 0), out=buf256)
        del arg243_1
        buf257 = reinterpret_tensor(buf256, (512, 1, 4096), (4096, 4096, 1), 0); del buf256  # reuse
        # Source Nodes: [output_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf257, arg244_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg244_1
        buf258 = reinterpret_tensor(buf251, (512, 1024), (1024, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg245_1, (4096, 1024), (1, 4096), 0), out=buf258)
        del arg245_1
        buf262 = reinterpret_tensor(buf238, (512, 1, 1024), (1024, 1024, 1), 0); del buf238  # reuse
        # Source Nodes: [add_59, cat_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf258, arg246_1, buf255, arg247_1, arg248_1, buf262, 512, 1024, grid=grid(512), stream=stream0)
        del arg246_1
        del arg247_1
        del arg248_1
        buf263 = reinterpret_tensor(buf258, (1, 512, 1024), (524288, 1024, 1), 0); del buf258  # reuse
        # Source Nodes: [q_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg70_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf263)
        del arg70_1
        buf264 = reinterpret_tensor(buf255, (1, 512, 1024), (524288, 1024, 1), 0); del buf255  # reuse
        # Source Nodes: [k_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg71_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf264)
        del arg71_1
        buf265 = reinterpret_tensor(buf237, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf237  # reuse
        buf268 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, add_61], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf263, arg74_1, arg75_1, buf265, buf268, 524288, grid=grid(524288), stream=stream0)
        del arg74_1
        del arg75_1
        buf266 = reinterpret_tensor(buf247, (16, 512, 512), (262144, 512, 1), 0); del buf247  # reuse
        # Source Nodes: [ac_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf264, (16, 64, 512), (64, 1, 1024), 0), out=buf266)
        buf267 = reinterpret_tensor(buf250, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf250  # reuse
        # Source Nodes: [k_head_r_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg73_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf267)
        del arg73_1
        buf269 = buf243; del buf243  # reuse
        # Source Nodes: [bd_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf267, (16, 64, 1024), (64, 1, 1024), 0), out=buf269)
        buf273 = reinterpret_tensor(buf240, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf240  # reuse
        # Source Nodes: [add_62, add_63, attn_prob_20, attn_score_10, bd_21], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf266, buf269, buf273, 8192, 512, grid=grid(8192), stream=stream0)
        buf272 = reinterpret_tensor(buf268, (1, 512, 1024), (524288, 1024, 1), 0); del buf268  # reuse
        # Source Nodes: [v_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg72_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf272)
        del arg72_1
        buf274 = reinterpret_tensor(buf265, (16, 512, 64), (32768, 64, 1), 0); del buf265  # reuse
        # Source Nodes: [attn_vec_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf273, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf272, (16, 512, 64), (64, 1024, 1), 0), out=buf274)
        buf275 = reinterpret_tensor(buf272, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf272  # reuse
        # Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf274, buf275, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf276 = reinterpret_tensor(buf267, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf267  # reuse
        # Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg76_1, buf276, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg76_1
        buf277 = reinterpret_tensor(buf274, (1, 512, 1024), (524288, 1024, 1), 0); del buf274  # reuse
        # Source Nodes: [attn_out_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf275, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf276, (1, 1024, 1024), (0, 1024, 1), 0), out=buf277)
        buf281 = reinterpret_tensor(buf275, (512, 1, 1024), (1024, 1024, 1), 0); del buf275  # reuse
        # Source Nodes: [attn_out_32, output_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf277, buf262, arg249_1, arg250_1, buf281, 512, 1024, grid=grid(512), stream=stream0)
        del arg249_1
        del arg250_1
        buf282 = reinterpret_tensor(buf257, (512, 4096), (4096, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 4096), (1, 1024), 0), out=buf282)
        del arg251_1
        buf283 = reinterpret_tensor(buf282, (512, 1, 4096), (4096, 4096, 1), 0); del buf282  # reuse
        # Source Nodes: [output_83], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf283, arg252_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg252_1
        buf284 = reinterpret_tensor(buf277, (512, 1024), (1024, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg253_1, (4096, 1024), (1, 4096), 0), out=buf284)
        del arg253_1
        buf288 = reinterpret_tensor(buf264, (512, 1, 1024), (1024, 1024, 1), 0); del buf264  # reuse
        # Source Nodes: [add_65, cat_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf284, arg254_1, buf281, arg255_1, arg256_1, buf288, 512, 1024, grid=grid(512), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        buf289 = reinterpret_tensor(buf284, (1, 512, 1024), (524288, 1024, 1), 0); del buf284  # reuse
        # Source Nodes: [q_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg77_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf289)
        del arg77_1
        buf290 = reinterpret_tensor(buf281, (1, 512, 1024), (524288, 1024, 1), 0); del buf281  # reuse
        # Source Nodes: [k_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg78_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf290)
        del arg78_1
        buf291 = reinterpret_tensor(buf263, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf263  # reuse
        buf294 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, add_67], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf289, arg81_1, arg82_1, buf291, buf294, 524288, grid=grid(524288), stream=stream0)
        del arg81_1
        del arg82_1
        buf292 = reinterpret_tensor(buf273, (16, 512, 512), (262144, 512, 1), 0); del buf273  # reuse
        # Source Nodes: [ac_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf290, (16, 64, 512), (64, 1, 1024), 0), out=buf292)
        buf293 = reinterpret_tensor(buf276, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf276  # reuse
        # Source Nodes: [k_head_r_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg80_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf293)
        del arg80_1
        buf295 = buf269; del buf269  # reuse
        # Source Nodes: [bd_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf293, (16, 64, 1024), (64, 1, 1024), 0), out=buf295)
        buf299 = reinterpret_tensor(buf266, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf266  # reuse
        # Source Nodes: [add_68, add_69, attn_prob_22, attn_score_11, bd_23], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf292, buf295, buf299, 8192, 512, grid=grid(8192), stream=stream0)
        buf298 = reinterpret_tensor(buf294, (1, 512, 1024), (524288, 1024, 1), 0); del buf294  # reuse
        # Source Nodes: [v_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg79_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf298)
        del arg79_1
        buf300 = reinterpret_tensor(buf291, (16, 512, 64), (32768, 64, 1), 0); del buf291  # reuse
        # Source Nodes: [attn_vec_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf299, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf298, (16, 512, 64), (64, 1024, 1), 0), out=buf300)
        buf301 = reinterpret_tensor(buf298, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf298  # reuse
        # Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf300, buf301, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf302 = reinterpret_tensor(buf293, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf293  # reuse
        # Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg83_1, buf302, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg83_1
        buf303 = reinterpret_tensor(buf300, (1, 512, 1024), (524288, 1024, 1), 0); del buf300  # reuse
        # Source Nodes: [attn_out_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf302, (1, 1024, 1024), (0, 1024, 1), 0), out=buf303)
        buf307 = reinterpret_tensor(buf301, (512, 1, 1024), (1024, 1024, 1), 0); del buf301  # reuse
        # Source Nodes: [attn_out_35, output_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf303, buf288, arg257_1, arg258_1, buf307, 512, 1024, grid=grid(512), stream=stream0)
        del arg257_1
        del arg258_1
        buf308 = reinterpret_tensor(buf283, (512, 4096), (4096, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg259_1, (1024, 4096), (1, 1024), 0), out=buf308)
        del arg259_1
        buf309 = reinterpret_tensor(buf308, (512, 1, 4096), (4096, 4096, 1), 0); del buf308  # reuse
        # Source Nodes: [output_91], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf309, arg260_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg260_1
        buf310 = reinterpret_tensor(buf303, (512, 1024), (1024, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf309, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg261_1, (4096, 1024), (1, 4096), 0), out=buf310)
        del arg261_1
        buf314 = reinterpret_tensor(buf290, (512, 1, 1024), (1024, 1024, 1), 0); del buf290  # reuse
        # Source Nodes: [add_71, cat_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf310, arg262_1, buf307, arg263_1, arg264_1, buf314, 512, 1024, grid=grid(512), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        buf315 = reinterpret_tensor(buf310, (1, 512, 1024), (524288, 1024, 1), 0); del buf310  # reuse
        # Source Nodes: [q_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg84_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf315)
        del arg84_1
        buf316 = reinterpret_tensor(buf307, (1, 512, 1024), (524288, 1024, 1), 0); del buf307  # reuse
        # Source Nodes: [k_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg85_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf316)
        del arg85_1
        buf317 = reinterpret_tensor(buf289, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf289  # reuse
        buf320 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_72, add_73], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf315, arg88_1, arg89_1, buf317, buf320, 524288, grid=grid(524288), stream=stream0)
        del arg88_1
        del arg89_1
        buf318 = reinterpret_tensor(buf299, (16, 512, 512), (262144, 512, 1), 0); del buf299  # reuse
        # Source Nodes: [ac_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf316, (16, 64, 512), (64, 1, 1024), 0), out=buf318)
        buf319 = reinterpret_tensor(buf302, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf302  # reuse
        # Source Nodes: [k_head_r_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg87_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf319)
        del arg87_1
        buf321 = buf295; del buf295  # reuse
        # Source Nodes: [bd_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf319, (16, 64, 1024), (64, 1, 1024), 0), out=buf321)
        buf325 = reinterpret_tensor(buf292, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf292  # reuse
        # Source Nodes: [add_74, add_75, attn_prob_24, attn_score_12, bd_25], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf318, buf321, buf325, 8192, 512, grid=grid(8192), stream=stream0)
        buf324 = reinterpret_tensor(buf320, (1, 512, 1024), (524288, 1024, 1), 0); del buf320  # reuse
        # Source Nodes: [v_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg86_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf324)
        del arg86_1
        buf326 = reinterpret_tensor(buf317, (16, 512, 64), (32768, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [attn_vec_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf324, (16, 512, 64), (64, 1024, 1), 0), out=buf326)
        buf327 = reinterpret_tensor(buf324, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf324  # reuse
        # Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf326, buf327, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf328 = reinterpret_tensor(buf319, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf319  # reuse
        # Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg90_1, buf328, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg90_1
        buf329 = reinterpret_tensor(buf326, (1, 512, 1024), (524288, 1024, 1), 0); del buf326  # reuse
        # Source Nodes: [attn_out_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf327, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf328, (1, 1024, 1024), (0, 1024, 1), 0), out=buf329)
        buf333 = reinterpret_tensor(buf327, (512, 1, 1024), (1024, 1024, 1), 0); del buf327  # reuse
        # Source Nodes: [attn_out_38, output_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf329, buf314, arg265_1, arg266_1, buf333, 512, 1024, grid=grid(512), stream=stream0)
        del arg265_1
        del arg266_1
        buf334 = reinterpret_tensor(buf309, (512, 4096), (4096, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 4096), (1, 1024), 0), out=buf334)
        del arg267_1
        buf335 = reinterpret_tensor(buf334, (512, 1, 4096), (4096, 4096, 1), 0); del buf334  # reuse
        # Source Nodes: [output_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf335, arg268_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg268_1
        buf336 = reinterpret_tensor(buf329, (512, 1024), (1024, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg269_1, (4096, 1024), (1, 4096), 0), out=buf336)
        del arg269_1
        buf340 = reinterpret_tensor(buf316, (512, 1, 1024), (1024, 1024, 1), 0); del buf316  # reuse
        # Source Nodes: [add_77, cat_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf336, arg270_1, buf333, arg271_1, arg272_1, buf340, 512, 1024, grid=grid(512), stream=stream0)
        del arg270_1
        del arg271_1
        del arg272_1
        buf341 = reinterpret_tensor(buf336, (1, 512, 1024), (524288, 1024, 1), 0); del buf336  # reuse
        # Source Nodes: [q_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg91_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf341)
        del arg91_1
        buf342 = reinterpret_tensor(buf333, (1, 512, 1024), (524288, 1024, 1), 0); del buf333  # reuse
        # Source Nodes: [k_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg92_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf342)
        del arg92_1
        buf343 = reinterpret_tensor(buf315, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf315  # reuse
        buf346 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_78, add_79], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf341, arg95_1, arg96_1, buf343, buf346, 524288, grid=grid(524288), stream=stream0)
        del arg95_1
        del arg96_1
        buf344 = reinterpret_tensor(buf325, (16, 512, 512), (262144, 512, 1), 0); del buf325  # reuse
        # Source Nodes: [ac_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf342, (16, 64, 512), (64, 1, 1024), 0), out=buf344)
        buf345 = reinterpret_tensor(buf328, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf328  # reuse
        # Source Nodes: [k_head_r_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg94_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf345)
        del arg94_1
        buf347 = buf321; del buf321  # reuse
        # Source Nodes: [bd_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf345, (16, 64, 1024), (64, 1, 1024), 0), out=buf347)
        buf351 = reinterpret_tensor(buf318, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf318  # reuse
        # Source Nodes: [add_80, add_81, attn_prob_26, attn_score_13, bd_27], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf344, buf347, buf351, 8192, 512, grid=grid(8192), stream=stream0)
        buf350 = reinterpret_tensor(buf346, (1, 512, 1024), (524288, 1024, 1), 0); del buf346  # reuse
        # Source Nodes: [v_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg93_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf350)
        del arg93_1
        buf352 = reinterpret_tensor(buf343, (16, 512, 64), (32768, 64, 1), 0); del buf343  # reuse
        # Source Nodes: [attn_vec_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf351, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf350, (16, 512, 64), (64, 1024, 1), 0), out=buf352)
        buf353 = reinterpret_tensor(buf350, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf350  # reuse
        # Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf352, buf353, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf354 = reinterpret_tensor(buf345, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf345  # reuse
        # Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg97_1, buf354, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg97_1
        buf355 = reinterpret_tensor(buf352, (1, 512, 1024), (524288, 1024, 1), 0); del buf352  # reuse
        # Source Nodes: [attn_out_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf354, (1, 1024, 1024), (0, 1024, 1), 0), out=buf355)
        buf359 = reinterpret_tensor(buf353, (512, 1, 1024), (1024, 1024, 1), 0); del buf353  # reuse
        # Source Nodes: [attn_out_41, output_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf355, buf340, arg273_1, arg274_1, buf359, 512, 1024, grid=grid(512), stream=stream0)
        del arg273_1
        del arg274_1
        buf360 = reinterpret_tensor(buf335, (512, 4096), (4096, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg275_1, (1024, 4096), (1, 1024), 0), out=buf360)
        del arg275_1
        buf361 = reinterpret_tensor(buf360, (512, 1, 4096), (4096, 4096, 1), 0); del buf360  # reuse
        # Source Nodes: [output_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf361, arg276_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg276_1
        buf362 = reinterpret_tensor(buf355, (512, 1024), (1024, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf361, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg277_1, (4096, 1024), (1, 4096), 0), out=buf362)
        del arg277_1
        buf366 = reinterpret_tensor(buf342, (512, 1, 1024), (1024, 1024, 1), 0); del buf342  # reuse
        # Source Nodes: [add_83, cat_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf362, arg278_1, buf359, arg279_1, arg280_1, buf366, 512, 1024, grid=grid(512), stream=stream0)
        del arg278_1
        del arg279_1
        del arg280_1
        buf367 = reinterpret_tensor(buf362, (1, 512, 1024), (524288, 1024, 1), 0); del buf362  # reuse
        # Source Nodes: [q_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg98_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf367)
        del arg98_1
        buf368 = reinterpret_tensor(buf359, (1, 512, 1024), (524288, 1024, 1), 0); del buf359  # reuse
        # Source Nodes: [k_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg99_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf368)
        del arg99_1
        buf369 = reinterpret_tensor(buf341, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf341  # reuse
        buf372 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_84, add_85], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf367, arg102_1, arg103_1, buf369, buf372, 524288, grid=grid(524288), stream=stream0)
        del arg102_1
        del arg103_1
        buf370 = reinterpret_tensor(buf351, (16, 512, 512), (262144, 512, 1), 0); del buf351  # reuse
        # Source Nodes: [ac_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf368, (16, 64, 512), (64, 1, 1024), 0), out=buf370)
        buf371 = reinterpret_tensor(buf354, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf354  # reuse
        # Source Nodes: [k_head_r_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg101_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf371)
        del arg101_1
        buf373 = buf347; del buf347  # reuse
        # Source Nodes: [bd_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf371, (16, 64, 1024), (64, 1, 1024), 0), out=buf373)
        buf377 = reinterpret_tensor(buf344, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf344  # reuse
        # Source Nodes: [add_86, add_87, attn_prob_28, attn_score_14, bd_29], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf370, buf373, buf377, 8192, 512, grid=grid(8192), stream=stream0)
        buf376 = reinterpret_tensor(buf372, (1, 512, 1024), (524288, 1024, 1), 0); del buf372  # reuse
        # Source Nodes: [v_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg100_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf376)
        del arg100_1
        buf378 = reinterpret_tensor(buf369, (16, 512, 64), (32768, 64, 1), 0); del buf369  # reuse
        # Source Nodes: [attn_vec_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf377, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf376, (16, 512, 64), (64, 1024, 1), 0), out=buf378)
        buf379 = reinterpret_tensor(buf376, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf376  # reuse
        # Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf378, buf379, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf380 = reinterpret_tensor(buf371, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf371  # reuse
        # Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg104_1, buf380, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg104_1
        buf381 = reinterpret_tensor(buf378, (1, 512, 1024), (524288, 1024, 1), 0); del buf378  # reuse
        # Source Nodes: [attn_out_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf379, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf380, (1, 1024, 1024), (0, 1024, 1), 0), out=buf381)
        buf385 = reinterpret_tensor(buf379, (512, 1, 1024), (1024, 1024, 1), 0); del buf379  # reuse
        # Source Nodes: [attn_out_44, output_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf381, buf366, arg281_1, arg282_1, buf385, 512, 1024, grid=grid(512), stream=stream0)
        del arg281_1
        del arg282_1
        buf386 = reinterpret_tensor(buf361, (512, 4096), (4096, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 4096), (1, 1024), 0), out=buf386)
        del arg283_1
        buf387 = reinterpret_tensor(buf386, (512, 1, 4096), (4096, 4096, 1), 0); del buf386  # reuse
        # Source Nodes: [output_115], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf387, arg284_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg284_1
        buf388 = reinterpret_tensor(buf381, (512, 1024), (1024, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg285_1, (4096, 1024), (1, 4096), 0), out=buf388)
        del arg285_1
        buf392 = reinterpret_tensor(buf368, (512, 1, 1024), (1024, 1024, 1), 0); del buf368  # reuse
        # Source Nodes: [add_89, cat_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf388, arg286_1, buf385, arg287_1, arg288_1, buf392, 512, 1024, grid=grid(512), stream=stream0)
        del arg286_1
        del arg287_1
        del arg288_1
        buf393 = reinterpret_tensor(buf388, (1, 512, 1024), (524288, 1024, 1), 0); del buf388  # reuse
        # Source Nodes: [q_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg105_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf393)
        del arg105_1
        buf394 = reinterpret_tensor(buf385, (1, 512, 1024), (524288, 1024, 1), 0); del buf385  # reuse
        # Source Nodes: [k_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg106_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf394)
        del arg106_1
        buf395 = reinterpret_tensor(buf367, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf367  # reuse
        buf398 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_90, add_91], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf393, arg109_1, arg110_1, buf395, buf398, 524288, grid=grid(524288), stream=stream0)
        del arg109_1
        del arg110_1
        buf396 = reinterpret_tensor(buf377, (16, 512, 512), (262144, 512, 1), 0); del buf377  # reuse
        # Source Nodes: [ac_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf394, (16, 64, 512), (64, 1, 1024), 0), out=buf396)
        buf397 = reinterpret_tensor(buf380, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf380  # reuse
        # Source Nodes: [k_head_r_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg108_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf397)
        del arg108_1
        buf399 = buf373; del buf373  # reuse
        # Source Nodes: [bd_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf398, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf397, (16, 64, 1024), (64, 1, 1024), 0), out=buf399)
        buf403 = reinterpret_tensor(buf370, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf370  # reuse
        # Source Nodes: [add_92, add_93, attn_prob_30, attn_score_15, bd_31], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf396, buf399, buf403, 8192, 512, grid=grid(8192), stream=stream0)
        buf402 = reinterpret_tensor(buf398, (1, 512, 1024), (524288, 1024, 1), 0); del buf398  # reuse
        # Source Nodes: [v_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg107_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf402)
        del arg107_1
        buf404 = reinterpret_tensor(buf395, (16, 512, 64), (32768, 64, 1), 0); del buf395  # reuse
        # Source Nodes: [attn_vec_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf402, (16, 512, 64), (64, 1024, 1), 0), out=buf404)
        buf405 = reinterpret_tensor(buf402, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf402  # reuse
        # Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf404, buf405, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf406 = reinterpret_tensor(buf397, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf397  # reuse
        # Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg111_1, buf406, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg111_1
        buf407 = reinterpret_tensor(buf404, (1, 512, 1024), (524288, 1024, 1), 0); del buf404  # reuse
        # Source Nodes: [attn_out_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf405, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf406, (1, 1024, 1024), (0, 1024, 1), 0), out=buf407)
        buf411 = reinterpret_tensor(buf405, (512, 1, 1024), (1024, 1024, 1), 0); del buf405  # reuse
        # Source Nodes: [attn_out_47, output_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf407, buf392, arg289_1, arg290_1, buf411, 512, 1024, grid=grid(512), stream=stream0)
        del arg289_1
        del arg290_1
        buf412 = reinterpret_tensor(buf387, (512, 4096), (4096, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg291_1, (1024, 4096), (1, 1024), 0), out=buf412)
        del arg291_1
        buf413 = reinterpret_tensor(buf412, (512, 1, 4096), (4096, 4096, 1), 0); del buf412  # reuse
        # Source Nodes: [output_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf413, arg292_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg292_1
        buf414 = reinterpret_tensor(buf407, (512, 1024), (1024, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf413, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg293_1, (4096, 1024), (1, 4096), 0), out=buf414)
        del arg293_1
        buf418 = reinterpret_tensor(buf394, (512, 1, 1024), (1024, 1024, 1), 0); del buf394  # reuse
        # Source Nodes: [add_95, cat_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf414, arg294_1, buf411, arg295_1, arg296_1, buf418, 512, 1024, grid=grid(512), stream=stream0)
        del arg294_1
        del arg295_1
        del arg296_1
        buf419 = reinterpret_tensor(buf414, (1, 512, 1024), (524288, 1024, 1), 0); del buf414  # reuse
        # Source Nodes: [q_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg112_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf419)
        del arg112_1
        buf420 = reinterpret_tensor(buf411, (1, 512, 1024), (524288, 1024, 1), 0); del buf411  # reuse
        # Source Nodes: [k_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg113_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf420)
        del arg113_1
        buf421 = reinterpret_tensor(buf393, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf393  # reuse
        buf424 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_96, add_97], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf419, arg116_1, arg117_1, buf421, buf424, 524288, grid=grid(524288), stream=stream0)
        del arg116_1
        del arg117_1
        buf422 = reinterpret_tensor(buf403, (16, 512, 512), (262144, 512, 1), 0); del buf403  # reuse
        # Source Nodes: [ac_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf420, (16, 64, 512), (64, 1, 1024), 0), out=buf422)
        buf423 = reinterpret_tensor(buf406, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf406  # reuse
        # Source Nodes: [k_head_r_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg115_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf423)
        del arg115_1
        buf425 = buf399; del buf399  # reuse
        # Source Nodes: [bd_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf423, (16, 64, 1024), (64, 1, 1024), 0), out=buf425)
        buf429 = reinterpret_tensor(buf396, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf396  # reuse
        # Source Nodes: [add_98, add_99, attn_prob_32, attn_score_16, bd_33], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf422, buf425, buf429, 8192, 512, grid=grid(8192), stream=stream0)
        buf428 = reinterpret_tensor(buf424, (1, 512, 1024), (524288, 1024, 1), 0); del buf424  # reuse
        # Source Nodes: [v_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf418, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg114_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf428)
        del arg114_1
        buf430 = reinterpret_tensor(buf421, (16, 512, 64), (32768, 64, 1), 0); del buf421  # reuse
        # Source Nodes: [attn_vec_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf429, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf428, (16, 512, 64), (64, 1024, 1), 0), out=buf430)
        buf431 = reinterpret_tensor(buf428, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf428  # reuse
        # Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf430, buf431, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf432 = reinterpret_tensor(buf423, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf423  # reuse
        # Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg118_1, buf432, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg118_1
        buf433 = reinterpret_tensor(buf430, (1, 512, 1024), (524288, 1024, 1), 0); del buf430  # reuse
        # Source Nodes: [attn_out_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf431, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf432, (1, 1024, 1024), (0, 1024, 1), 0), out=buf433)
        buf437 = reinterpret_tensor(buf431, (512, 1, 1024), (1024, 1024, 1), 0); del buf431  # reuse
        # Source Nodes: [attn_out_50, output_129], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf433, buf418, arg297_1, arg298_1, buf437, 512, 1024, grid=grid(512), stream=stream0)
        del arg297_1
        del arg298_1
        buf438 = reinterpret_tensor(buf413, (512, 4096), (4096, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf437, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 4096), (1, 1024), 0), out=buf438)
        del arg299_1
        buf439 = reinterpret_tensor(buf438, (512, 1, 4096), (4096, 4096, 1), 0); del buf438  # reuse
        # Source Nodes: [output_131], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf439, arg300_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg300_1
        buf440 = reinterpret_tensor(buf433, (512, 1024), (1024, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg301_1, (4096, 1024), (1, 4096), 0), out=buf440)
        del arg301_1
        buf444 = reinterpret_tensor(buf420, (512, 1, 1024), (1024, 1024, 1), 0); del buf420  # reuse
        # Source Nodes: [add_101, cat_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf440, arg302_1, buf437, arg303_1, arg304_1, buf444, 512, 1024, grid=grid(512), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        buf445 = reinterpret_tensor(buf440, (1, 512, 1024), (524288, 1024, 1), 0); del buf440  # reuse
        # Source Nodes: [q_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg119_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf445)
        del arg119_1
        buf446 = reinterpret_tensor(buf437, (1, 512, 1024), (524288, 1024, 1), 0); del buf437  # reuse
        # Source Nodes: [k_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg120_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf446)
        del arg120_1
        buf447 = reinterpret_tensor(buf419, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf419  # reuse
        buf450 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_102, add_103], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf445, arg123_1, arg124_1, buf447, buf450, 524288, grid=grid(524288), stream=stream0)
        del arg123_1
        del arg124_1
        buf448 = reinterpret_tensor(buf429, (16, 512, 512), (262144, 512, 1), 0); del buf429  # reuse
        # Source Nodes: [ac_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf446, (16, 64, 512), (64, 1, 1024), 0), out=buf448)
        buf449 = reinterpret_tensor(buf432, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf432  # reuse
        # Source Nodes: [k_head_r_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg122_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf449)
        del arg122_1
        buf451 = buf425; del buf425  # reuse
        # Source Nodes: [bd_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf449, (16, 64, 1024), (64, 1, 1024), 0), out=buf451)
        buf455 = reinterpret_tensor(buf422, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf422  # reuse
        # Source Nodes: [add_104, add_105, attn_prob_34, attn_score_17, bd_35], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf448, buf451, buf455, 8192, 512, grid=grid(8192), stream=stream0)
        buf454 = reinterpret_tensor(buf450, (1, 512, 1024), (524288, 1024, 1), 0); del buf450  # reuse
        # Source Nodes: [v_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg121_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf454)
        del arg121_1
        buf456 = reinterpret_tensor(buf447, (16, 512, 64), (32768, 64, 1), 0); del buf447  # reuse
        # Source Nodes: [attn_vec_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf455, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf454, (16, 512, 64), (64, 1024, 1), 0), out=buf456)
        buf457 = reinterpret_tensor(buf454, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf454  # reuse
        # Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf456, buf457, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf458 = reinterpret_tensor(buf449, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf449  # reuse
        # Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg125_1, buf458, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg125_1
        buf459 = reinterpret_tensor(buf456, (1, 512, 1024), (524288, 1024, 1), 0); del buf456  # reuse
        # Source Nodes: [attn_out_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf457, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf458, (1, 1024, 1024), (0, 1024, 1), 0), out=buf459)
        buf463 = reinterpret_tensor(buf457, (512, 1, 1024), (1024, 1024, 1), 0); del buf457  # reuse
        # Source Nodes: [attn_out_53, output_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf459, buf444, arg305_1, arg306_1, buf463, 512, 1024, grid=grid(512), stream=stream0)
        del arg305_1
        del arg306_1
        buf464 = reinterpret_tensor(buf439, (512, 4096), (4096, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg307_1, (1024, 4096), (1, 1024), 0), out=buf464)
        del arg307_1
        buf465 = reinterpret_tensor(buf464, (512, 1, 4096), (4096, 4096, 1), 0); del buf464  # reuse
        # Source Nodes: [output_139], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf465, arg308_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg308_1
        buf466 = reinterpret_tensor(buf459, (512, 1024), (1024, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg309_1, (4096, 1024), (1, 4096), 0), out=buf466)
        del arg309_1
        buf470 = reinterpret_tensor(buf446, (512, 1, 1024), (1024, 1024, 1), 0); del buf446  # reuse
        # Source Nodes: [add_107, cat_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf466, arg310_1, buf463, arg311_1, arg312_1, buf470, 512, 1024, grid=grid(512), stream=stream0)
        del arg310_1
        del arg311_1
        del arg312_1
        buf471 = reinterpret_tensor(buf466, (1, 512, 1024), (524288, 1024, 1), 0); del buf466  # reuse
        # Source Nodes: [q_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg126_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf471)
        del arg126_1
        buf472 = reinterpret_tensor(buf463, (1, 512, 1024), (524288, 1024, 1), 0); del buf463  # reuse
        # Source Nodes: [k_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg127_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf472)
        del arg127_1
        buf473 = reinterpret_tensor(buf445, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf445  # reuse
        buf476 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_108, add_109], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf471, arg130_1, arg131_1, buf473, buf476, 524288, grid=grid(524288), stream=stream0)
        del arg130_1
        del arg131_1
        buf474 = reinterpret_tensor(buf455, (16, 512, 512), (262144, 512, 1), 0); del buf455  # reuse
        # Source Nodes: [ac_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf472, (16, 64, 512), (64, 1, 1024), 0), out=buf474)
        buf475 = reinterpret_tensor(buf458, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf458  # reuse
        # Source Nodes: [k_head_r_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg129_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf475)
        del arg129_1
        buf477 = buf451; del buf451  # reuse
        # Source Nodes: [bd_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf476, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf475, (16, 64, 1024), (64, 1, 1024), 0), out=buf477)
        buf481 = reinterpret_tensor(buf448, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf448  # reuse
        # Source Nodes: [add_110, add_111, attn_prob_36, attn_score_18, bd_37], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf474, buf477, buf481, 8192, 512, grid=grid(8192), stream=stream0)
        buf480 = reinterpret_tensor(buf476, (1, 512, 1024), (524288, 1024, 1), 0); del buf476  # reuse
        # Source Nodes: [v_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg128_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf480)
        del arg128_1
        buf482 = reinterpret_tensor(buf473, (16, 512, 64), (32768, 64, 1), 0); del buf473  # reuse
        # Source Nodes: [attn_vec_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf481, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf480, (16, 512, 64), (64, 1024, 1), 0), out=buf482)
        buf483 = reinterpret_tensor(buf480, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf480  # reuse
        # Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf482, buf483, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf484 = reinterpret_tensor(buf475, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf475  # reuse
        # Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg132_1, buf484, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg132_1
        buf485 = reinterpret_tensor(buf482, (1, 512, 1024), (524288, 1024, 1), 0); del buf482  # reuse
        # Source Nodes: [attn_out_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf483, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf484, (1, 1024, 1024), (0, 1024, 1), 0), out=buf485)
        buf489 = reinterpret_tensor(buf483, (512, 1, 1024), (1024, 1024, 1), 0); del buf483  # reuse
        # Source Nodes: [attn_out_56, output_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf485, buf470, arg313_1, arg314_1, buf489, 512, 1024, grid=grid(512), stream=stream0)
        del arg313_1
        del arg314_1
        buf490 = reinterpret_tensor(buf465, (512, 4096), (4096, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf489, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 4096), (1, 1024), 0), out=buf490)
        del arg315_1
        buf491 = reinterpret_tensor(buf490, (512, 1, 4096), (4096, 4096, 1), 0); del buf490  # reuse
        # Source Nodes: [output_147], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf491, arg316_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg316_1
        buf492 = reinterpret_tensor(buf485, (512, 1024), (1024, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf491, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg317_1, (4096, 1024), (1, 4096), 0), out=buf492)
        del arg317_1
        buf496 = reinterpret_tensor(buf472, (512, 1, 1024), (1024, 1024, 1), 0); del buf472  # reuse
        # Source Nodes: [add_113, cat_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf492, arg318_1, buf489, arg319_1, arg320_1, buf496, 512, 1024, grid=grid(512), stream=stream0)
        del arg318_1
        del arg319_1
        del arg320_1
        buf497 = reinterpret_tensor(buf492, (1, 512, 1024), (524288, 1024, 1), 0); del buf492  # reuse
        # Source Nodes: [q_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg133_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf497)
        del arg133_1
        buf498 = reinterpret_tensor(buf489, (1, 512, 1024), (524288, 1024, 1), 0); del buf489  # reuse
        # Source Nodes: [k_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg134_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf498)
        del arg134_1
        buf499 = reinterpret_tensor(buf471, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf471  # reuse
        buf502 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_114, add_115], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf497, arg137_1, arg138_1, buf499, buf502, 524288, grid=grid(524288), stream=stream0)
        del arg137_1
        del arg138_1
        buf500 = reinterpret_tensor(buf481, (16, 512, 512), (262144, 512, 1), 0); del buf481  # reuse
        # Source Nodes: [ac_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf498, (16, 64, 512), (64, 1, 1024), 0), out=buf500)
        buf501 = reinterpret_tensor(buf484, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf484  # reuse
        # Source Nodes: [k_head_r_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg136_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf501)
        del arg136_1
        buf503 = buf477; del buf477  # reuse
        # Source Nodes: [bd_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf501, (16, 64, 1024), (64, 1, 1024), 0), out=buf503)
        buf507 = reinterpret_tensor(buf474, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf474  # reuse
        # Source Nodes: [add_116, add_117, attn_prob_38, attn_score_19, bd_39], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf500, buf503, buf507, 8192, 512, grid=grid(8192), stream=stream0)
        buf506 = reinterpret_tensor(buf502, (1, 512, 1024), (524288, 1024, 1), 0); del buf502  # reuse
        # Source Nodes: [v_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf496, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg135_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf506)
        del arg135_1
        buf508 = reinterpret_tensor(buf499, (16, 512, 64), (32768, 64, 1), 0); del buf499  # reuse
        # Source Nodes: [attn_vec_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf507, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf506, (16, 512, 64), (64, 1024, 1), 0), out=buf508)
        buf509 = reinterpret_tensor(buf506, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf506  # reuse
        # Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf508, buf509, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf510 = reinterpret_tensor(buf501, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf501  # reuse
        # Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg139_1, buf510, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg139_1
        buf511 = reinterpret_tensor(buf508, (1, 512, 1024), (524288, 1024, 1), 0); del buf508  # reuse
        # Source Nodes: [attn_out_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf509, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf510, (1, 1024, 1024), (0, 1024, 1), 0), out=buf511)
        buf515 = reinterpret_tensor(buf509, (512, 1, 1024), (1024, 1024, 1), 0); del buf509  # reuse
        # Source Nodes: [attn_out_59, output_153], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf511, buf496, arg321_1, arg322_1, buf515, 512, 1024, grid=grid(512), stream=stream0)
        del arg321_1
        del arg322_1
        buf516 = reinterpret_tensor(buf491, (512, 4096), (4096, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg323_1, (1024, 4096), (1, 1024), 0), out=buf516)
        del arg323_1
        buf517 = reinterpret_tensor(buf516, (512, 1, 4096), (4096, 4096, 1), 0); del buf516  # reuse
        # Source Nodes: [output_155], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf517, arg324_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg324_1
        buf518 = reinterpret_tensor(buf511, (512, 1024), (1024, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg325_1, (4096, 1024), (1, 4096), 0), out=buf518)
        del arg325_1
        buf522 = reinterpret_tensor(buf498, (512, 1, 1024), (1024, 1024, 1), 0); del buf498  # reuse
        # Source Nodes: [add_119, cat_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf518, arg326_1, buf515, arg327_1, arg328_1, buf522, 512, 1024, grid=grid(512), stream=stream0)
        del arg326_1
        del arg327_1
        del arg328_1
        buf523 = reinterpret_tensor(buf518, (1, 512, 1024), (524288, 1024, 1), 0); del buf518  # reuse
        # Source Nodes: [q_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg140_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf523)
        del arg140_1
        buf524 = reinterpret_tensor(buf515, (1, 512, 1024), (524288, 1024, 1), 0); del buf515  # reuse
        # Source Nodes: [k_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg141_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf524)
        del arg141_1
        buf525 = reinterpret_tensor(buf497, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf497  # reuse
        buf528 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_120, add_121], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf523, arg144_1, arg145_1, buf525, buf528, 524288, grid=grid(524288), stream=stream0)
        del arg144_1
        del arg145_1
        buf526 = reinterpret_tensor(buf507, (16, 512, 512), (262144, 512, 1), 0); del buf507  # reuse
        # Source Nodes: [ac_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf524, (16, 64, 512), (64, 1, 1024), 0), out=buf526)
        buf527 = reinterpret_tensor(buf510, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf510  # reuse
        # Source Nodes: [k_head_r_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg143_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf527)
        del arg143_1
        buf529 = buf503; del buf503  # reuse
        # Source Nodes: [bd_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf528, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf527, (16, 64, 1024), (64, 1, 1024), 0), out=buf529)
        buf533 = reinterpret_tensor(buf500, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf500  # reuse
        # Source Nodes: [add_122, add_123, attn_prob_40, attn_score_20, bd_41], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf526, buf529, buf533, 8192, 512, grid=grid(8192), stream=stream0)
        buf532 = reinterpret_tensor(buf528, (1, 512, 1024), (524288, 1024, 1), 0); del buf528  # reuse
        # Source Nodes: [v_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg142_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf532)
        del arg142_1
        buf534 = reinterpret_tensor(buf525, (16, 512, 64), (32768, 64, 1), 0); del buf525  # reuse
        # Source Nodes: [attn_vec_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf533, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf532, (16, 512, 64), (64, 1024, 1), 0), out=buf534)
        buf535 = reinterpret_tensor(buf532, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf532  # reuse
        # Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf534, buf535, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf536 = reinterpret_tensor(buf527, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf527  # reuse
        # Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg146_1, buf536, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg146_1
        buf537 = reinterpret_tensor(buf534, (1, 512, 1024), (524288, 1024, 1), 0); del buf534  # reuse
        # Source Nodes: [attn_out_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf535, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf536, (1, 1024, 1024), (0, 1024, 1), 0), out=buf537)
        buf541 = reinterpret_tensor(buf535, (512, 1, 1024), (1024, 1024, 1), 0); del buf535  # reuse
        # Source Nodes: [attn_out_62, output_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf537, buf522, arg329_1, arg330_1, buf541, 512, 1024, grid=grid(512), stream=stream0)
        del arg329_1
        del arg330_1
        buf542 = reinterpret_tensor(buf517, (512, 4096), (4096, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf541, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 4096), (1, 1024), 0), out=buf542)
        del arg331_1
        buf543 = reinterpret_tensor(buf542, (512, 1, 4096), (4096, 4096, 1), 0); del buf542  # reuse
        # Source Nodes: [output_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf543, arg332_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg332_1
        buf544 = reinterpret_tensor(buf537, (512, 1024), (1024, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf543, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg333_1, (4096, 1024), (1, 4096), 0), out=buf544)
        del arg333_1
        buf548 = reinterpret_tensor(buf524, (512, 1, 1024), (1024, 1024, 1), 0); del buf524  # reuse
        # Source Nodes: [add_125, cat_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf544, arg334_1, buf541, arg335_1, arg336_1, buf548, 512, 1024, grid=grid(512), stream=stream0)
        del arg334_1
        del arg335_1
        del arg336_1
        buf549 = reinterpret_tensor(buf544, (1, 512, 1024), (524288, 1024, 1), 0); del buf544  # reuse
        # Source Nodes: [q_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg147_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf549)
        del arg147_1
        buf550 = reinterpret_tensor(buf541, (1, 512, 1024), (524288, 1024, 1), 0); del buf541  # reuse
        # Source Nodes: [k_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg148_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf550)
        del arg148_1
        buf551 = reinterpret_tensor(buf523, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf523  # reuse
        buf554 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, add_127], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf549, arg151_1, arg152_1, buf551, buf554, 524288, grid=grid(524288), stream=stream0)
        del arg151_1
        del arg152_1
        buf552 = reinterpret_tensor(buf533, (16, 512, 512), (262144, 512, 1), 0); del buf533  # reuse
        # Source Nodes: [ac_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf551, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf550, (16, 64, 512), (64, 1, 1024), 0), out=buf552)
        buf553 = reinterpret_tensor(buf536, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf536  # reuse
        # Source Nodes: [k_head_r_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg150_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf553)
        del arg150_1
        buf555 = buf529; del buf529  # reuse
        # Source Nodes: [bd_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf554, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf553, (16, 64, 1024), (64, 1, 1024), 0), out=buf555)
        buf559 = reinterpret_tensor(buf526, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf526  # reuse
        # Source Nodes: [add_128, add_129, attn_prob_42, attn_score_21, bd_43], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf552, buf555, buf559, 8192, 512, grid=grid(8192), stream=stream0)
        buf558 = reinterpret_tensor(buf554, (1, 512, 1024), (524288, 1024, 1), 0); del buf554  # reuse
        # Source Nodes: [v_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg149_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf558)
        del arg149_1
        buf560 = reinterpret_tensor(buf551, (16, 512, 64), (32768, 64, 1), 0); del buf551  # reuse
        # Source Nodes: [attn_vec_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf559, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf558, (16, 512, 64), (64, 1024, 1), 0), out=buf560)
        buf561 = reinterpret_tensor(buf558, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf558  # reuse
        # Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf560, buf561, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf562 = reinterpret_tensor(buf553, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf553  # reuse
        # Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg153_1, buf562, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg153_1
        buf563 = reinterpret_tensor(buf560, (1, 512, 1024), (524288, 1024, 1), 0); del buf560  # reuse
        # Source Nodes: [attn_out_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf561, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf562, (1, 1024, 1024), (0, 1024, 1), 0), out=buf563)
        buf567 = reinterpret_tensor(buf561, (512, 1, 1024), (1024, 1024, 1), 0); del buf561  # reuse
        # Source Nodes: [attn_out_65, output_169], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf563, buf548, arg337_1, arg338_1, buf567, 512, 1024, grid=grid(512), stream=stream0)
        del arg337_1
        del arg338_1
        buf568 = reinterpret_tensor(buf543, (512, 4096), (4096, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg339_1, (1024, 4096), (1, 1024), 0), out=buf568)
        del arg339_1
        buf569 = reinterpret_tensor(buf568, (512, 1, 4096), (4096, 4096, 1), 0); del buf568  # reuse
        # Source Nodes: [output_171], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf569, arg340_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg340_1
        buf570 = reinterpret_tensor(buf563, (512, 1024), (1024, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf569, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg341_1, (4096, 1024), (1, 4096), 0), out=buf570)
        del arg341_1
        buf574 = reinterpret_tensor(buf550, (512, 1, 1024), (1024, 1024, 1), 0); del buf550  # reuse
        # Source Nodes: [add_131, cat_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf570, arg342_1, buf567, arg343_1, arg344_1, buf574, 512, 1024, grid=grid(512), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        buf575 = reinterpret_tensor(buf570, (1, 512, 1024), (524288, 1024, 1), 0); del buf570  # reuse
        # Source Nodes: [q_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg154_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf575)
        del arg154_1
        buf576 = reinterpret_tensor(buf567, (1, 512, 1024), (524288, 1024, 1), 0); del buf567  # reuse
        # Source Nodes: [k_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg155_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf576)
        del arg155_1
        buf577 = reinterpret_tensor(buf549, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf549  # reuse
        buf580 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_132, add_133], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf575, arg158_1, arg159_1, buf577, buf580, 524288, grid=grid(524288), stream=stream0)
        del arg158_1
        del arg159_1
        buf578 = reinterpret_tensor(buf559, (16, 512, 512), (262144, 512, 1), 0); del buf559  # reuse
        # Source Nodes: [ac_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf576, (16, 64, 512), (64, 1, 1024), 0), out=buf578)
        buf579 = reinterpret_tensor(buf562, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf562  # reuse
        # Source Nodes: [k_head_r_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg157_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf579)
        del arg157_1
        buf581 = buf555; del buf555  # reuse
        # Source Nodes: [bd_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf580, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf579, (16, 64, 1024), (64, 1, 1024), 0), out=buf581)
        buf585 = reinterpret_tensor(buf552, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf552  # reuse
        # Source Nodes: [add_134, add_135, attn_prob_44, attn_score_22, bd_45], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf578, buf581, buf585, 8192, 512, grid=grid(8192), stream=stream0)
        buf584 = reinterpret_tensor(buf580, (1, 512, 1024), (524288, 1024, 1), 0); del buf580  # reuse
        # Source Nodes: [v_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg156_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf584)
        del arg156_1
        buf586 = reinterpret_tensor(buf577, (16, 512, 64), (32768, 64, 1), 0); del buf577  # reuse
        # Source Nodes: [attn_vec_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf584, (16, 512, 64), (64, 1024, 1), 0), out=buf586)
        buf587 = reinterpret_tensor(buf584, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf584  # reuse
        # Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf586, buf587, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf588 = reinterpret_tensor(buf579, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf579  # reuse
        # Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg160_1, buf588, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg160_1
        buf589 = reinterpret_tensor(buf586, (1, 512, 1024), (524288, 1024, 1), 0); del buf586  # reuse
        # Source Nodes: [attn_out_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf587, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf588, (1, 1024, 1024), (0, 1024, 1), 0), out=buf589)
        buf593 = reinterpret_tensor(buf587, (512, 1, 1024), (1024, 1024, 1), 0); del buf587  # reuse
        # Source Nodes: [attn_out_68, output_177], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf589, buf574, arg345_1, arg346_1, buf593, 512, 1024, grid=grid(512), stream=stream0)
        del arg345_1
        del arg346_1
        buf594 = reinterpret_tensor(buf569, (512, 4096), (4096, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf593, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 4096), (1, 1024), 0), out=buf594)
        del arg347_1
        buf595 = reinterpret_tensor(buf594, (512, 1, 4096), (4096, 4096, 1), 0); del buf594  # reuse
        # Source Nodes: [output_179], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf595, arg348_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg348_1
        buf596 = reinterpret_tensor(buf589, (512, 1024), (1024, 1), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf595, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg349_1, (4096, 1024), (1, 4096), 0), out=buf596)
        del arg349_1
        buf600 = reinterpret_tensor(buf576, (512, 1, 1024), (1024, 1024, 1), 0); del buf576  # reuse
        # Source Nodes: [add_137, cat_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf596, arg350_1, buf593, arg351_1, arg352_1, buf600, 512, 1024, grid=grid(512), stream=stream0)
        del arg350_1
        del arg351_1
        del arg352_1
        buf601 = reinterpret_tensor(buf596, (1, 512, 1024), (524288, 1024, 1), 0); del buf596  # reuse
        # Source Nodes: [q_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf600, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg161_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf601)
        del arg161_1
        buf602 = reinterpret_tensor(buf593, (1, 512, 1024), (524288, 1024, 1), 0); del buf593  # reuse
        # Source Nodes: [k_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf600, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg162_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf602)
        del arg162_1
        buf603 = reinterpret_tensor(buf575, (512, 1, 16, 64), (1024, 1024, 64, 1), 0); del buf575  # reuse
        buf606 = empty((512, 1, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_138, add_139], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf601, arg165_1, arg166_1, buf603, buf606, 524288, grid=grid(524288), stream=stream0)
        del arg165_1
        del arg166_1
        del buf601
        buf604 = reinterpret_tensor(buf585, (16, 512, 512), (262144, 512, 1), 0); del buf585  # reuse
        # Source Nodes: [ac_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf603, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf602, (16, 64, 512), (64, 1, 1024), 0), out=buf604)
        buf605 = reinterpret_tensor(buf588, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf588  # reuse
        # Source Nodes: [k_head_r_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(arg164_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf605)
        del arg164_1
        del buf6
        buf607 = buf581; del buf581  # reuse
        # Source Nodes: [bd_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf606, (16, 512, 64), (64, 1024, 1), 0), reinterpret_tensor(buf605, (16, 64, 1024), (64, 1, 1024), 0), out=buf607)
        buf611 = reinterpret_tensor(buf578, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf578  # reuse
        # Source Nodes: [add_140, add_141, attn_prob_46, attn_score_23, bd_47], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
        triton_red_fused__softmax_add_index_select_mul_3.run(buf604, buf607, buf611, 8192, 512, grid=grid(8192), stream=stream0)
        del buf604
        del buf607
        buf610 = reinterpret_tensor(buf606, (1, 512, 1024), (524288, 1024, 1), 0); del buf606  # reuse
        # Source Nodes: [v_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf600, (1, 512, 1024), (524288, 1024, 1), 0), reinterpret_tensor(arg163_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf610)
        del arg163_1
        buf612 = reinterpret_tensor(buf603, (16, 512, 64), (32768, 64, 1), 0); del buf603  # reuse
        # Source Nodes: [attn_vec_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf611, (16, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf610, (16, 512, 64), (64, 1024, 1), 0), out=buf612)
        del buf611
        buf613 = reinterpret_tensor(buf610, (512, 64, 16, 1, 1), (1024, 16, 1, 1, 1), 0); del buf610  # reuse
        # Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf612, buf613, 32768, 16, grid=grid(32768, 16), stream=stream0)
        buf614 = reinterpret_tensor(buf605, (64, 16, 1, 1024, 1), (16384, 1024, 1024, 1, 1), 0); del buf605  # reuse
        # Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(arg167_1, buf614, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg167_1
        buf615 = reinterpret_tensor(buf612, (1, 512, 1024), (524288, 1024, 1), 0); del buf612  # reuse
        # Source Nodes: [attn_out_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf613, (1, 512, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf614, (1, 1024, 1024), (0, 1024, 1), 0), out=buf615)
        del buf614
        buf619 = reinterpret_tensor(buf613, (512, 1, 1024), (1024, 1024, 1), 0); del buf613  # reuse
        # Source Nodes: [attn_out_71, output_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf615, buf600, arg353_1, arg354_1, buf619, 512, 1024, grid=grid(512), stream=stream0)
        del arg353_1
        del arg354_1
        buf620 = reinterpret_tensor(buf595, (512, 4096), (4096, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf619, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg355_1, (1024, 4096), (1, 1024), 0), out=buf620)
        del arg355_1
        buf621 = reinterpret_tensor(buf620, (512, 1, 4096), (4096, 4096, 1), 0); del buf620  # reuse
        # Source Nodes: [output_187], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf621, arg356_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg356_1
        buf622 = reinterpret_tensor(buf615, (512, 1024), (1024, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf621, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg357_1, (4096, 1024), (1, 4096), 0), out=buf622)
        del arg357_1
        del buf621
        buf626 = reinterpret_tensor(buf602, (512, 1, 1024), (1024, 1024, 1), 0); del buf602  # reuse
        # Source Nodes: [add_143, output_h_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf622, arg358_1, buf619, arg359_1, arg360_1, buf626, 512, 1024, grid=grid(512), stream=stream0)
        del arg358_1
        del arg359_1
        del arg360_1
        del buf619
        del buf622
        buf627 = empty((512, 32000), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg362_1, reinterpret_tensor(buf626, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 32000), (1, 1024), 0), alpha=1, beta=1, out=buf627)
        del arg361_1
        del arg362_1
        del buf626
        buf628 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        buf629 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_9.run(buf627, buf628, buf629, 512, 32000, grid=grid(512), stream=stream0)
        buf630 = empty((), device='cuda', dtype=torch.float32)
        buf632 = buf630; del buf630  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_10.run(buf632, arg364_1, buf627, buf628, buf629, 1, 512, grid=grid(1), stream=stream0)
        del arg364_1
        return (buf632, reinterpret_tensor(buf627, (1, 512, 32000), (16384000, 32000, 1), 0), buf0, buf28, buf54, buf80, buf106, buf132, buf158, buf184, buf210, buf236, buf262, buf288, buf314, buf340, buf366, buf392, buf418, buf444, buf470, buf496, buf522, buf548, buf574, buf600, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg364_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
