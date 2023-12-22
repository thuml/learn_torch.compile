
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


cpp_fused_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (144L*x1) + (432L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (432L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(57600L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x2 + (57600L*x1) + (172800L*x0))];
                        out_ptr2[static_cast<long>(x1 + (3L*x2) + (172800L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_index_add_mul_sub_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.0);
            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
            auto tmp11 = static_cast<float>(-0.75);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(-3.75);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = static_cast<float>(-6.0);
            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
            auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
            auto tmp19 = static_cast<float>(-3.0);
            auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
            out_ptr0[static_cast<long>(x0)] = tmp20;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.25);
            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
            auto tmp11 = static_cast<float>(2.25);
            auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
            auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
            auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
            auto tmp15 = static_cast<float>(1.0);
            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
            out_ptr1[static_cast<long>(x0)] = tmp16;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.0);
            auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
            auto tmp11 = static_cast<float>(1.25);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(2.25);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
            auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
            out_ptr2[static_cast<long>(x0)] = tmp17;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(2.0);
            auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
            auto tmp11 = static_cast<float>(-0.75);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(-3.75);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = static_cast<float>(-6.0);
            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
            auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
            auto tmp19 = static_cast<float>(-3.0);
            auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
            out_ptr3[static_cast<long>(x0)] = tmp20;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.0);
            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
            auto tmp11 = static_cast<float>(-0.75);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(-3.75);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = static_cast<float>(-6.0);
            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
            auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
            auto tmp19 = static_cast<float>(-3.0);
            auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
            out_ptr4[static_cast<long>(x0)] = tmp20;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.25);
            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
            auto tmp11 = static_cast<float>(2.25);
            auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
            auto tmp13 = decltype(tmp12)(tmp12 * tmp8);
            auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
            auto tmp15 = static_cast<float>(1.0);
            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
            out_ptr5[static_cast<long>(x0)] = tmp16;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(1.0);
            auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
            auto tmp11 = static_cast<float>(1.25);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(2.25);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
            auto tmp17 = decltype(tmp16)(tmp16 + tmp9);
            out_ptr6[static_cast<long>(x0)] = tmp17;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = c10::convert<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = static_cast<float>(1.0714285714285714);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
            auto tmp7 = std::floor(tmp6);
            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
            auto tmp9 = static_cast<float>(2.0);
            auto tmp10 = decltype(tmp9)(tmp9 - tmp8);
            auto tmp11 = static_cast<float>(-0.75);
            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
            auto tmp13 = static_cast<float>(-3.75);
            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
            auto tmp15 = decltype(tmp14)(tmp14 * tmp10);
            auto tmp16 = static_cast<float>(-6.0);
            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
            auto tmp18 = decltype(tmp17)(tmp17 * tmp10);
            auto tmp19 = static_cast<float>(-3.0);
            auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
            out_ptr7[static_cast<long>(x0)] = tmp20;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(224L); x3+=static_cast<long>(1L))
                        {
                            auto tmp25 = out_ptr0[static_cast<long>(x3)];
                            auto tmp30 = out_ptr1[static_cast<long>(x3)];
                            auto tmp37 = out_ptr2[static_cast<long>(x3)];
                            auto tmp45 = out_ptr3[static_cast<long>(x3)];
                            auto tmp90 = out_ptr4[static_cast<long>(x2)];
                            auto tmp92 = out_ptr5[static_cast<long>(x2)];
                            auto tmp95 = out_ptr6[static_cast<long>(x2)];
                            auto tmp98 = out_ptr7[static_cast<long>(x2)];
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<float>(tmp0);
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                            auto tmp4 = static_cast<float>(1.0714285714285714);
                            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                            auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                            auto tmp7 = std::floor(tmp6);
                            auto tmp8 = c10::convert<long>(tmp7);
                            auto tmp9 = static_cast<long>(0);
                            auto tmp10 = max_propagate_nan(tmp8, tmp9);
                            auto tmp11 = static_cast<long>(239);
                            auto tmp12 = min_propagate_nan(tmp10, tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 + tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp4);
                            auto tmp17 = decltype(tmp16)(tmp16 - tmp2);
                            auto tmp18 = std::floor(tmp17);
                            auto tmp19 = c10::convert<long>(tmp18);
                            auto tmp20 = static_cast<long>(1);
                            auto tmp21 = decltype(tmp19)(tmp19 - tmp20);
                            auto tmp22 = max_propagate_nan(tmp21, tmp9);
                            auto tmp23 = min_propagate_nan(tmp22, tmp11);
                            auto tmp24 = in_ptr0[static_cast<long>(x1 + (3L*tmp23) + (720L*tmp12) + (172800L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = max_propagate_nan(tmp19, tmp9);
                            auto tmp28 = min_propagate_nan(tmp27, tmp11);
                            auto tmp29 = in_ptr0[static_cast<long>(x1 + (3L*tmp28) + (720L*tmp12) + (172800L*x0))];
                            auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                            auto tmp32 = decltype(tmp26)(tmp26 + tmp31);
                            auto tmp33 = decltype(tmp19)(tmp19 + tmp20);
                            auto tmp34 = max_propagate_nan(tmp33, tmp9);
                            auto tmp35 = min_propagate_nan(tmp34, tmp11);
                            auto tmp36 = in_ptr0[static_cast<long>(x1 + (3L*tmp35) + (720L*tmp12) + (172800L*x0))];
                            auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                            auto tmp39 = decltype(tmp32)(tmp32 + tmp38);
                            auto tmp40 = static_cast<long>(2);
                            auto tmp41 = decltype(tmp19)(tmp19 + tmp40);
                            auto tmp42 = max_propagate_nan(tmp41, tmp9);
                            auto tmp43 = min_propagate_nan(tmp42, tmp11);
                            auto tmp44 = in_ptr0[static_cast<long>(x1 + (3L*tmp43) + (720L*tmp12) + (172800L*x0))];
                            auto tmp46 = decltype(tmp44)(tmp44 * tmp45);
                            auto tmp47 = decltype(tmp39)(tmp39 + tmp46);
                            auto tmp48 = decltype(tmp8)(tmp8 - tmp20);
                            auto tmp49 = max_propagate_nan(tmp48, tmp9);
                            auto tmp50 = min_propagate_nan(tmp49, tmp11);
                            auto tmp51 = in_ptr0[static_cast<long>(x1 + (3L*tmp23) + (720L*tmp50) + (172800L*x0))];
                            auto tmp52 = decltype(tmp51)(tmp51 * tmp25);
                            auto tmp53 = in_ptr0[static_cast<long>(x1 + (3L*tmp28) + (720L*tmp50) + (172800L*x0))];
                            auto tmp54 = decltype(tmp53)(tmp53 * tmp30);
                            auto tmp55 = decltype(tmp52)(tmp52 + tmp54);
                            auto tmp56 = decltype(tmp8)(tmp8 + tmp20);
                            auto tmp57 = max_propagate_nan(tmp56, tmp9);
                            auto tmp58 = min_propagate_nan(tmp57, tmp11);
                            auto tmp59 = in_ptr0[static_cast<long>(x1 + (3L*tmp23) + (720L*tmp58) + (172800L*x0))];
                            auto tmp60 = decltype(tmp59)(tmp59 * tmp25);
                            auto tmp61 = in_ptr0[static_cast<long>(x1 + (3L*tmp28) + (720L*tmp58) + (172800L*x0))];
                            auto tmp62 = decltype(tmp61)(tmp61 * tmp30);
                            auto tmp63 = decltype(tmp60)(tmp60 + tmp62);
                            auto tmp64 = decltype(tmp8)(tmp8 + tmp40);
                            auto tmp65 = max_propagate_nan(tmp64, tmp9);
                            auto tmp66 = min_propagate_nan(tmp65, tmp11);
                            auto tmp67 = in_ptr0[static_cast<long>(x1 + (3L*tmp23) + (720L*tmp66) + (172800L*x0))];
                            auto tmp68 = decltype(tmp67)(tmp67 * tmp25);
                            auto tmp69 = in_ptr0[static_cast<long>(x1 + (3L*tmp28) + (720L*tmp66) + (172800L*x0))];
                            auto tmp70 = decltype(tmp69)(tmp69 * tmp30);
                            auto tmp71 = decltype(tmp68)(tmp68 + tmp70);
                            auto tmp72 = in_ptr0[static_cast<long>(x1 + (3L*tmp35) + (720L*tmp50) + (172800L*x0))];
                            auto tmp73 = decltype(tmp72)(tmp72 * tmp37);
                            auto tmp74 = decltype(tmp55)(tmp55 + tmp73);
                            auto tmp75 = in_ptr0[static_cast<long>(x1 + (3L*tmp43) + (720L*tmp50) + (172800L*x0))];
                            auto tmp76 = decltype(tmp75)(tmp75 * tmp45);
                            auto tmp77 = decltype(tmp74)(tmp74 + tmp76);
                            auto tmp78 = in_ptr0[static_cast<long>(x1 + (3L*tmp35) + (720L*tmp58) + (172800L*x0))];
                            auto tmp79 = decltype(tmp78)(tmp78 * tmp37);
                            auto tmp80 = decltype(tmp63)(tmp63 + tmp79);
                            auto tmp81 = in_ptr0[static_cast<long>(x1 + (3L*tmp43) + (720L*tmp58) + (172800L*x0))];
                            auto tmp82 = decltype(tmp81)(tmp81 * tmp45);
                            auto tmp83 = decltype(tmp80)(tmp80 + tmp82);
                            auto tmp84 = in_ptr0[static_cast<long>(x1 + (3L*tmp35) + (720L*tmp66) + (172800L*x0))];
                            auto tmp85 = decltype(tmp84)(tmp84 * tmp37);
                            auto tmp86 = decltype(tmp71)(tmp71 + tmp85);
                            auto tmp87 = in_ptr0[static_cast<long>(x1 + (3L*tmp43) + (720L*tmp66) + (172800L*x0))];
                            auto tmp88 = decltype(tmp87)(tmp87 * tmp45);
                            auto tmp89 = decltype(tmp86)(tmp86 + tmp88);
                            auto tmp91 = decltype(tmp77)(tmp77 * tmp90);
                            auto tmp93 = decltype(tmp47)(tmp47 * tmp92);
                            auto tmp94 = decltype(tmp91)(tmp91 + tmp93);
                            auto tmp96 = decltype(tmp83)(tmp83 * tmp95);
                            auto tmp97 = decltype(tmp94)(tmp94 + tmp96);
                            auto tmp99 = decltype(tmp89)(tmp89 * tmp98);
                            auto tmp100 = decltype(tmp97)(tmp97 + tmp99);
                            out_ptr12[static_cast<long>(x1 + (3L*x3) + (672L*x2) + (150528L*x0))] = tmp100;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        tmp28.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(401);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp16;
                            auto tmp28 = tmp27 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (401L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (401L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(128.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        tmp30.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(400L))) + (51200L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (256L*x1)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr4 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp24 = decltype(tmp20)::blendv(tmp23, tmp20, tmp14);
                            auto tmp25 = tmp24 + tmp16;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp25);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(256.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        tmp28.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp14 = to_float_mask(tmp4);
                            auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                            auto tmp17 = tmp15 + tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp14);
                            auto tmp27 = tmp26 + tmp16;
                            auto tmp28 = tmp27 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(256.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        tmp30.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (50176L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(403456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp19 = static_cast<float>(0.7071067811865476);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp15 * tmp20;
                        auto tmp22 = tmp21.erf();
                        auto tmp23 = static_cast<float>(1.0);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 + tmp24;
                        auto tmp26 = tmp18 * tmp25;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        tmp26.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(256.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = static_cast<float>(0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = static_cast<float>(0.7071067811865476);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp15 * tmp20;
                auto tmp22 = tmp21.erf();
                auto tmp23 = static_cast<float>(1.0);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp22 + tmp24;
                auto tmp26 = tmp18 * tmp25;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp26.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(256.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        auto tmp16 = [&]
                        {
                            auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp17;
                        }
                        ;
                        auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp4));
                        auto tmp19 = [&]
                        {
                            auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            return tmp20;
                        }
                        ;
                        auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp8));
                        auto tmp22 = decltype(tmp18)::blendv(tmp21, tmp18, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp22.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(128.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            return tmp12;
                        }
                        ;
                        auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp14 = to_float_mask(tmp4);
                        auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                        tmp15.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(403456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp5 = out_ptr0[static_cast<long>(x0)];
                auto tmp8 = out_ptr1[static_cast<long>(x0)];
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 - tmp6;
                auto tmp9 = static_cast<float>(128.0);
                auto tmp10 = tmp8 / tmp9;
                auto tmp11 = static_cast<float>(1e-06);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp13 = 1 / std::sqrt(tmp12);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp7 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                auto tmp20 = static_cast<float>(0.5);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = static_cast<float>(0.7071067811865476);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp19 * tmp24;
                auto tmp26 = tmp25.erf();
                auto tmp27 = static_cast<float>(1.0);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp26 + tmp28;
                auto tmp30 = tmp22 * tmp29;
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp30.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp5 = out_ptr0[static_cast<long>(x0)];
                auto tmp8 = out_ptr1[static_cast<long>(x0)];
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 - tmp6;
                auto tmp9 = static_cast<float>(256.0);
                auto tmp10 = tmp8 / tmp9;
                auto tmp11 = static_cast<float>(1e-06);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp13 = 1 / std::sqrt(tmp12);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp7 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                auto tmp20 = static_cast<float>(0.5);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = static_cast<float>(0.7071067811865476);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp19 * tmp24;
                auto tmp26 = tmp25.erf();
                auto tmp27 = static_cast<float>(1.0);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp26 + tmp28;
                auto tmp30 = tmp22 * tmp29;
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp30.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(256.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp21;
                        }
                        ;
                        auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                        auto tmp23 = [&]
                        {
                            auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp25 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp26 = tmp24 + tmp25;
                            auto tmp27 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp28 = tmp26 + tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                        auto tmp30 = decltype(tmp22)::blendv(tmp29, tmp22, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp30.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(128.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(403456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp5 = out_ptr0[static_cast<long>(x0)];
                auto tmp8 = out_ptr1[static_cast<long>(x0)];
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 - tmp6;
                auto tmp9 = static_cast<float>(128.0);
                auto tmp10 = tmp8 / tmp9;
                auto tmp11 = static_cast<float>(1e-06);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp13 = 1 / std::sqrt(tmp12);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp7 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                auto tmp20 = static_cast<float>(0.5);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = static_cast<float>(0.7071067811865476);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp19 * tmp24;
                auto tmp26 = tmp25.erf();
                auto tmp27 = static_cast<float>(1.0);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp26 + tmp28;
                auto tmp30 = tmp22 * tmp29;
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp30.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp5 = out_ptr0[static_cast<long>(x0)];
                auto tmp8 = out_ptr1[static_cast<long>(x0)];
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 - tmp6;
                auto tmp9 = static_cast<float>(256.0);
                auto tmp10 = tmp8 / tmp9;
                auto tmp11 = static_cast<float>(1e-06);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp13 = 1 / std::sqrt(tmp12);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp7 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                auto tmp20 = static_cast<float>(0.5);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = static_cast<float>(0.7071067811865476);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp19 * tmp24;
                auto tmp26 = tmp25.erf();
                auto tmp27 = static_cast<float>(1.0);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp26 + tmp28;
                auto tmp30 = tmp22 * tmp29;
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp30.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50432L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (50432L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50432L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (50432L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(256.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(401);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = masked_load(in_ptr4 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp4));
                            return tmp21;
                        }
                        ;
                        auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                        auto tmp23 = [&]
                        {
                            auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp25 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp26 = tmp24 + tmp25;
                            auto tmp27 = masked_load(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp8));
                            auto tmp28 = tmp26 + tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                        auto tmp30 = decltype(tmp22)::blendv(tmp29, tmp22, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp30.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(400L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(400L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (51328L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (401L*x1) + (401L*x1_inner) + (51328L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (401L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (401L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr2 + static_cast<long>(x1 + (401L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(400L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (401L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                out_ptr2[static_cast<long>(x1 + (401L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x1) + (128L*x2) + (51328L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (12832L*x1) + (51328L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_native_layer_norm_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (51328L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr1[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(128.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-06);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = static_cast<float>(0.5);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp17 * tmp19;
                auto tmp21 = static_cast<float>(0.7071067811865476);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                auto tmp24 = tmp23.erf();
                auto tmp25 = static_cast<float>(1.0);
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 + tmp26;
                auto tmp28 = tmp20 * tmp27;
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp28.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_clone_native_layer_norm_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr2 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (51328L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(401L*x0)];
                        auto tmp4 = in_out_ptr0[static_cast<long>(401L*x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (50432L*x0)));
                        auto tmp1 = out_ptr3[static_cast<long>(197L*x0)];
                        auto tmp4 = in_out_ptr1[static_cast<long>(197L*x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(out_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_cat_detach_mean_native_layer_norm_native_layer_norm_backward_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1000L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = c10::convert<int>(x0);
                auto tmp1 = static_cast<int>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<int>(8);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (1000L*x0)), to_float_mask(tmp4));
                    return tmp6;
                }
                ;
                auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<int>(16);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-8000L) + x1 + (1000L*x0)), to_float_mask(tmp8));
                    return tmp12;
                }
                ;
                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                auto tmp14 = to_float_mask(tmp4);
                auto tmp15 = decltype(tmp7)::blendv(tmp13, tmp7, tmp14);
                auto tmp16 = c10::convert<int>(8L + x0);
                auto tmp17 = tmp16 >= tmp1;
                auto tmp18 = tmp16 < tmp3;
                auto tmp19 = [&]
                {
                    auto tmp20 = masked_load(in_ptr0 + static_cast<long>(8000L + x1 + (1000L*x0)), to_float_mask(tmp18));
                    return tmp20;
                }
                ;
                auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                auto tmp22 = tmp16 >= tmp3;
                auto tmp23 = tmp16 < tmp9;
                auto tmp24 = [&]
                {
                    auto tmp25 = masked_load(in_ptr1 + static_cast<long>(x1 + (1000L*x0)), to_float_mask(tmp22));
                    return tmp25;
                }
                ;
                auto tmp26 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp22));
                auto tmp27 = to_float_mask(tmp18);
                auto tmp28 = decltype(tmp21)::blendv(tmp26, tmp21, tmp27);
                auto tmp29 = tmp15 + tmp28;
                auto tmp30 = static_cast<float>(2.0);
                auto tmp31 = at::vec::Vectorized<float>(tmp30);
                auto tmp32 = tmp29 / tmp31;
                tmp32.store(out_ptr0 + static_cast<long>(x1 + (1000L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))] = tmp2;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (4L*x2) + (788L*x0))] = tmp2;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr4 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr3[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr4[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr8[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr5[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (32L*x1) + (128L*x0))];
                        out_ptr6[static_cast<long>(x1 + (4L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr10[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                            auto tmp1 = in_ptr11[static_cast<long>(x1 + (4L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr7[static_cast<long>(x1 + (4L*x2) + (1604L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr11 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr12[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                            auto tmp1 = in_ptr13[static_cast<long>(x1 + (4L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr8[static_cast<long>(x1 + (4L*x2) + (788L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr14[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr9[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr15[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr10[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr16[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr11[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr17[static_cast<long>(x2 + (32L*x1) + (128L*x0))];
                        out_ptr12[static_cast<long>(x1 + (4L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr18[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                            auto tmp1 = in_ptr19[static_cast<long>(x1 + (4L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr13[static_cast<long>(x1 + (4L*x2) + (1604L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr20[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                            auto tmp1 = in_ptr21[static_cast<long>(x1 + (4L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr14[static_cast<long>(x1 + (4L*x2) + (788L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr22[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr15[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr23[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr16[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr24[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr17[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr25[static_cast<long>(x2 + (32L*x1) + (128L*x0))];
                        out_ptr18[static_cast<long>(x1 + (4L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr31 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_2, (1, 401, 128), (51328, 128, 1))
    assert_size_stride(primals_3, (1, 1, 256), (256, 256, 1))
    assert_size_stride(primals_4, (1, 197, 256), (50432, 256, 1))
    assert_size_stride(primals_5, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (384, 128), (128, 1))
    assert_size_stride(primals_12, (384, ), (1, ))
    assert_size_stride(primals_13, (128, 128), (128, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (384, 128), (128, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (128, 384), (384, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (768, 256), (256, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (256, 256), (256, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (768, 256), (256, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (256, 768), (768, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (768, 256), (256, 1))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (256, 256), (256, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (768, 256), (256, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (256, 768), (768, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (768, 256), (256, 1))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (256, 256), (256, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (768, 256), (256, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (256, 768), (768, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (256, 128), (128, 1))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (128, 256), (256, 1))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 256), (256, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, 256), (256, 1))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, 256), (256, 1))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, 256), (256, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (128, 256), (256, 1))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, 128), (128, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, 128), (128, 1))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, 128), (128, 1))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, 128), (128, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (256, 128), (128, 1))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (384, 128), (128, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (128, 128), (128, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (384, 128), (128, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (128, 384), (384, 1))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (768, 256), (256, 1))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (256, 256), (256, 1))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (768, 256), (256, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (256, 768), (768, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (768, 256), (256, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (256, 256), (256, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (768, 256), (256, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (256, 768), (768, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (768, 256), (256, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (256, 256), (256, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (768, 256), (256, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (256, 768), (768, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (256, 128), (128, 1))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (128, 256), (256, 1))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, 256), (256, 1))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, 256), (256, 1))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (256, 256), (256, 1))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (256, 256), (256, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (128, 256), (256, 1))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, 128), (128, 1))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, 128), (128, 1))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, 128), (128, 1))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, 128), (128, 1))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (256, 128), (128, 1))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (384, 128), (128, 1))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (128, 128), (128, 1))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (384, 128), (128, 1))
    assert_size_stride(primals_186, (384, ), (1, ))
    assert_size_stride(primals_187, (128, 384), (384, 1))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (768, 256), (256, 1))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (256, 256), (256, 1))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (768, 256), (256, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (256, 768), (768, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (256, ), (1, ))
    assert_size_stride(primals_203, (768, 256), (256, 1))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (256, 256), (256, 1))
    assert_size_stride(primals_206, (256, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (768, 256), (256, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (256, 768), (768, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (768, 256), (256, 1))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (256, 256), (256, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (768, 256), (256, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (256, 768), (768, 1))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (256, 128), (128, 1))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (128, 256), (256, 1))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, 256), (256, 1))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, 256), (256, 1))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, 256), (256, 1))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256), (256, 1))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (128, 256), (256, 1))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (128, 128), (128, 1))
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, 128), (128, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, 128), (128, 1))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, 128), (128, 1))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_259, (256, 128), (128, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (1000, 128), (128, 1))
    assert_size_stride(primals_266, (1000, ), (1, ))
    assert_size_stride(primals_267, (1000, 256), (256, 1))
    assert_size_stride(primals_268, (1000, ), (1, ))
    assert_size_stride(primals_269, (8, 3, 240, 240), (172800, 57600, 240, 1))
    buf0 = empty_strided((128, 3, 12, 12), (432, 1, 36, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((256, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((8, 3, 240, 240), (172800, 1, 720, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_5.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()))
    del primals_269
    del primals_5
    del primals_7
    # Source Nodes: [l__mod___patch_embed_0_proj], Original ATen: [aten.convolution]
    buf3 = extern_kernels.convolution(buf2, buf0, primals_6, stride=(12, 12), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf3, (8, 128, 20, 20), (51200, 1, 2560, 128))
    del primals_6
    buf5 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf11 = empty((1, 1, 1, 224), device='cpu', dtype=torch.float32)
    buf21 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf23 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf25 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 1, 224, 1), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_index_add_mul_sub_1(c_void_p(buf2.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del buf11
    del buf21
    del buf23
    del buf25
    del buf27
    del buf5
    del buf7
    del buf9
    # Source Nodes: [l__mod___patch_embed_1_proj], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, buf1, primals_8, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf29, (8, 256, 14, 14), (50176, 1, 3584, 256))
    del primals_8
    buf30 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf33 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf34 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_2(c_void_p(primals_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del primals_10
    buf35 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf34, reinterpret_tensor(primals_11, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf35)
    del primals_12
    # Source Nodes: [x_3], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf36 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf37 = buf36[0]
    buf38 = buf36[1]
    buf39 = buf36[2]
    buf40 = buf36[3]
    buf41 = buf36[6]
    buf42 = buf36[7]
    del buf36
    buf44 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, reinterpret_tensor(buf37, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_13, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf44)
    del primals_14
    buf45 = buf30; del buf30  # reuse
    buf46 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf48 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf49 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_3(c_void_p(primals_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_16
    buf50 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf49, reinterpret_tensor(primals_17, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf50)
    del primals_18
    buf51 = empty((3208, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_4(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    buf52 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf51, reinterpret_tensor(primals_19, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf52)
    del primals_20
    buf53 = reinterpret_tensor(buf52, (8, 401, 128), (51328, 128, 1), 0); del buf52  # reuse
    buf54 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf57 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf58 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_5(c_void_p(buf53.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del buf3
    del primals_1
    del primals_2
    del primals_22
    buf59 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf58, reinterpret_tensor(primals_23, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf59)
    del primals_24
    # Source Nodes: [x_15], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf60 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf61 = buf60[0]
    buf62 = buf60[1]
    buf63 = buf60[2]
    buf64 = buf60[3]
    buf65 = buf60[6]
    buf66 = buf60[7]
    del buf60
    buf68 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, reinterpret_tensor(buf61, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_25, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf68)
    del primals_26
    buf69 = buf54; del buf54  # reuse
    buf70 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf72 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf73 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_6(c_void_p(primals_3.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_28
    buf74 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_30, buf73, reinterpret_tensor(primals_29, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf74)
    del primals_30
    buf75 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_7(c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf75, reinterpret_tensor(primals_31, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf76)
    del primals_32
    buf77 = reinterpret_tensor(buf76, (8, 197, 256), (50432, 256, 1), 0); del buf76  # reuse
    buf78 = buf69; del buf69  # reuse
    buf79 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf81 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf82 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_8(c_void_p(buf77.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del buf29
    del primals_3
    del primals_34
    del primals_4
    buf83 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_36, buf82, reinterpret_tensor(primals_35, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf83)
    del primals_36
    # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf84 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf85 = buf84[0]
    buf86 = buf84[1]
    buf87 = buf84[2]
    buf88 = buf84[3]
    buf89 = buf84[6]
    buf90 = buf84[7]
    del buf84
    buf92 = buf68; del buf68  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, reinterpret_tensor(buf85, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_37, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf92)
    del primals_38
    buf93 = buf78; del buf78  # reuse
    buf94 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf97 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_9(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_40
    buf98 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, buf97, reinterpret_tensor(primals_41, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf98)
    del primals_42
    buf99 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_10(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf99, reinterpret_tensor(primals_43, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf100)
    del primals_44
    buf101 = buf93; del buf93  # reuse
    buf102 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf104 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf105 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_46
    buf106 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf105, reinterpret_tensor(primals_47, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf106)
    del primals_48
    # Source Nodes: [x_39], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf107 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf108 = buf107[0]
    buf109 = buf107[1]
    buf110 = buf107[2]
    buf111 = buf107[3]
    buf112 = buf107[6]
    buf113 = buf107[7]
    del buf107
    buf115 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, reinterpret_tensor(buf108, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_49, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf115)
    del primals_50
    buf116 = buf101; del buf101  # reuse
    buf117 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf119 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf120 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_12(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    del primals_52
    buf121 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf120, reinterpret_tensor(primals_53, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf121)
    del primals_54
    buf122 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_13(c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf122, reinterpret_tensor(primals_55, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf123)
    del primals_56
    buf124 = reinterpret_tensor(buf123, (8, 197, 256), (50432, 256, 1), 0); del buf123  # reuse
    buf125 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf128 = empty((8, 1, 128), device='cpu', dtype=torch.float32)
    buf129 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_14(c_void_p(buf124.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf129, reinterpret_tensor(primals_59, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf130)
    del primals_60
    buf131 = buf125; del buf125  # reuse
    buf132 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf134 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf135 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_15(c_void_p(buf124.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_64, buf135, reinterpret_tensor(primals_63, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf136)
    del primals_64
    buf137 = reinterpret_tensor(buf92, (8, 197, 256), (50432, 256, 1), 0); del buf92  # reuse
    buf138 = reinterpret_tensor(buf116, (8, 197, 1), (197, 1, 1), 0); del buf116  # reuse
    buf139 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf141 = reinterpret_tensor(buf139, (8, 197, 1), (197, 1, 1), 0); del buf139  # reuse
    buf142 = buf77; del buf77  # reuse
    cpp_fused_cat_native_layer_norm_16(c_void_p(buf141.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf142.data_ptr()))
    del primals_66
    buf143 = buf130; del buf130  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_67, (256, 256), (1, 256), 0), out=buf143)
    buf144 = buf115; del buf115  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, reinterpret_tensor(buf142, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf144)
    del primals_70
    buf145 = buf100; del buf100  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, reinterpret_tensor(buf142, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_71, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf145)
    del primals_72
    buf146 = reinterpret_tensor(buf143, (8, 1, 256), (256, 256, 1), 0); del buf143  # reuse
    buf147 = empty((8, 4, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_17(c_void_p(buf146.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_68
    buf148 = empty((32, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf146, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf147, (32, 64, 197), (12608, 197, 1), 0), out=buf148)
    buf149 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf150 = reinterpret_tensor(buf148, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf148  # reuse
    buf151 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf152 = empty((8, 4, 1, 197), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf144, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf144  # reuse
    cpp_fused__softmax_clone_mul_18(c_void_p(buf150.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    buf154 = empty((32, 1, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf153, (32, 197, 64), (12608, 64, 1), 0), out=buf154)
    buf155 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, reinterpret_tensor(buf154, (8, 256), (256, 1), 0), reinterpret_tensor(primals_73, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf155)
    del primals_74
    buf156 = buf131; del buf131  # reuse
    buf157 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf159 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf160 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_19(c_void_p(buf137.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [reverted_proj_cls_token], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf160, reinterpret_tensor(primals_77, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf161)
    del primals_78
    buf162 = reinterpret_tensor(buf44, (8, 401, 128), (51328, 128, 1), 0); del buf44  # reuse
    buf163 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf45, (8, 401, 1), (401, 1, 1), 0); del buf45  # reuse
    buf165 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf165, (8, 401, 1), (401, 1, 1), 0); del buf165  # reuse
    buf168 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_20(c_void_p(buf167.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()))
    del primals_80
    buf169 = buf161; del buf161  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_81, (128, 128), (1, 128), 0), out=buf169)
    buf170 = reinterpret_tensor(buf53, (3208, 128), (128, 1), 0); del buf53  # reuse
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, reinterpret_tensor(buf168, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_83, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf170)
    del primals_84
    buf171 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, reinterpret_tensor(buf168, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_85, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf171)
    del primals_86
    buf172 = reinterpret_tensor(buf169, (8, 1, 128), (128, 128, 1), 0); del buf169  # reuse
    buf173 = empty((8, 4, 32, 401), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_21(c_void_p(buf172.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_82
    buf174 = empty((32, 1, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf173, (32, 32, 401), (12832, 401, 1), 0), out=buf174)
    buf175 = buf149; del buf149  # reuse
    buf176 = reinterpret_tensor(buf174, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf174  # reuse
    buf177 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf178 = empty((8, 4, 1, 401), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf170, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf170  # reuse
    cpp_fused__softmax_clone_mul_22(c_void_p(buf176.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = reinterpret_tensor(buf136, (32, 1, 32), (32, 32, 1), 0); del buf136  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf178, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf179, (32, 401, 32), (12832, 32, 1), 0), out=buf180)
    buf181 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, reinterpret_tensor(buf180, (8, 128), (128, 1), 0), reinterpret_tensor(primals_87, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf181)
    del primals_88
    buf182 = buf156; del buf156  # reuse
    buf183 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf185 = empty((8, 1, 128), device='cpu', dtype=torch.float32)
    buf186 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_23(c_void_p(buf163.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = buf155; del buf155  # reuse
    # Source Nodes: [reverted_proj_cls_token_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf186, reinterpret_tensor(primals_91, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf187)
    del primals_92
    buf188 = reinterpret_tensor(buf145, (8, 197, 256), (50432, 256, 1), 0); del buf145  # reuse
    buf189 = empty((8, 401, 1), device='cpu', dtype=torch.float32)
    buf190 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf192 = reinterpret_tensor(buf190, (8, 401, 1), (401, 1, 1), 0); del buf190  # reuse
    buf193 = buf171; del buf171  # reuse
    cpp_fused_cat_native_layer_norm_view_24(c_void_p(buf192.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_94
    buf194 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf193, reinterpret_tensor(primals_95, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf194)
    del primals_96
    # Source Nodes: [x_59], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf195 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf196 = buf195[0]
    buf197 = buf195[1]
    buf198 = buf195[2]
    buf199 = buf195[3]
    buf200 = buf195[6]
    buf201 = buf195[7]
    del buf195
    buf203 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, reinterpret_tensor(buf196, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_97, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf203)
    del primals_98
    buf204 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf207 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf208 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf162.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del primals_100
    buf209 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf208, reinterpret_tensor(primals_101, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf209)
    del primals_102
    buf210 = empty((3208, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_26(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf210, reinterpret_tensor(primals_103, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf211)
    del primals_104
    buf212 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf213 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf213, (8, 197, 1), (197, 1, 1), 0); del buf213  # reuse
    buf216 = reinterpret_tensor(buf124, (1576, 256), (256, 1), 0); del buf124  # reuse
    cpp_fused_native_layer_norm_view_27(c_void_p(buf215.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_106
    buf217 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf216, reinterpret_tensor(primals_107, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf217)
    del primals_108
    # Source Nodes: [x_71], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf218 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf219 = buf218[0]
    buf220 = buf218[1]
    buf221 = buf218[2]
    buf222 = buf218[3]
    buf223 = buf218[6]
    buf224 = buf218[7]
    del buf218
    buf226 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, reinterpret_tensor(buf219, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_109, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf226)
    del primals_110
    buf227 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf230 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf231 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf188.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_112
    buf232 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf231, reinterpret_tensor(primals_113, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf232)
    del primals_114
    buf233 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_29(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf233, reinterpret_tensor(primals_115, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf234)
    del primals_116
    buf235 = buf227; del buf227  # reuse
    buf236 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf238 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf239 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf188.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del primals_118
    buf240 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf239, reinterpret_tensor(primals_119, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf240)
    del primals_120
    # Source Nodes: [x_83], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf241 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf242 = buf241[0]
    buf243 = buf241[1]
    buf244 = buf241[2]
    buf245 = buf241[3]
    buf246 = buf241[6]
    buf247 = buf241[7]
    del buf241
    buf249 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, reinterpret_tensor(buf242, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_121, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf249)
    del primals_122
    buf250 = buf235; del buf235  # reuse
    buf251 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf253 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf254 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf188.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_124
    buf255 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf254, reinterpret_tensor(primals_125, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf255)
    del primals_126
    buf256 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_32(c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf256, reinterpret_tensor(primals_127, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf257)
    del primals_128
    buf258 = reinterpret_tensor(buf257, (8, 197, 256), (50432, 256, 1), 0); del buf257  # reuse
    buf259 = buf250; del buf250  # reuse
    buf260 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf262 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf263 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_33(c_void_p(buf258.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    del primals_130
    buf264 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf263, reinterpret_tensor(primals_131, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf264)
    del primals_132
    # Source Nodes: [x_95], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf265 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf266 = buf265[0]
    buf267 = buf265[1]
    buf268 = buf265[2]
    buf269 = buf265[3]
    buf270 = buf265[6]
    buf271 = buf265[7]
    del buf265
    buf273 = buf249; del buf249  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, reinterpret_tensor(buf266, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_133, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf273)
    del primals_134
    buf274 = buf259; del buf259  # reuse
    buf275 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf277 = reinterpret_tensor(buf234, (8, 197, 256), (50432, 256, 1), 0); del buf234  # reuse
    buf278 = buf226; del buf226  # reuse
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf258.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del primals_136
    buf279 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf278, reinterpret_tensor(primals_137, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf279)
    del primals_138
    buf280 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_35(c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf280, reinterpret_tensor(primals_139, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf281)
    del primals_140
    buf282 = buf182; del buf182  # reuse
    buf283 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf285 = reinterpret_tensor(buf181, (8, 1, 128), (128, 128, 1), 0); del buf181  # reuse
    buf286 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_36(c_void_p(buf162.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = buf187; del buf187  # reuse
    # Source Nodes: [l__mod___blocks_1_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf286, reinterpret_tensor(primals_143, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf287)
    del primals_144
    buf288 = buf282; del buf282  # reuse
    buf289 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf291 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf292 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_37(c_void_p(buf258.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    buf293 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf292, reinterpret_tensor(primals_147, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf293)
    del primals_148
    buf294 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf295 = reinterpret_tensor(buf274, (8, 197, 1), (197, 1, 1), 0); del buf274  # reuse
    buf296 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf298 = reinterpret_tensor(buf296, (8, 197, 1), (197, 1, 1), 0); del buf296  # reuse
    buf299 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_38(c_void_p(buf298.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf299.data_ptr()))
    del primals_150
    buf300 = buf287; del buf287  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_151, (256, 256), (1, 256), 0), out=buf300)
    buf301 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, reinterpret_tensor(buf299, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_153, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf301)
    del primals_154
    buf302 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, reinterpret_tensor(buf299, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_155, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf302)
    del primals_156
    buf303 = reinterpret_tensor(buf300, (8, 1, 256), (256, 256, 1), 0); del buf300  # reuse
    buf304 = empty((8, 4, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_39(c_void_p(buf303.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf304.data_ptr()))
    del primals_152
    buf305 = empty((32, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf303, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf304, (32, 64, 197), (12608, 197, 1), 0), out=buf305)
    buf306 = buf175; del buf175  # reuse
    buf307 = reinterpret_tensor(buf305, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf305  # reuse
    buf308 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf309 = empty((8, 4, 1, 197), device='cpu', dtype=torch.float32)
    buf310 = reinterpret_tensor(buf301, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf301  # reuse
    cpp_fused__softmax_clone_mul_40(c_void_p(buf307.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    buf311 = empty((32, 1, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf309, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf310, (32, 197, 64), (12608, 64, 1), 0), out=buf311)
    buf312 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, reinterpret_tensor(buf311, (8, 256), (256, 1), 0), reinterpret_tensor(primals_157, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf312)
    del primals_158
    buf313 = buf288; del buf288  # reuse
    buf314 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf316 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf317 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_41(c_void_p(buf294.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [reverted_proj_cls_token_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf317, reinterpret_tensor(primals_161, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf318)
    del primals_162
    buf319 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf320 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf321 = reinterpret_tensor(buf204, (8, 401, 1), (401, 1, 1), 0); del buf204  # reuse
    buf322 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf324 = reinterpret_tensor(buf322, (8, 401, 1), (401, 1, 1), 0); del buf322  # reuse
    buf325 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_42(c_void_p(buf324.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_164
    buf326 = buf318; del buf318  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_165, (128, 128), (1, 128), 0), out=buf326)
    buf327 = buf211; del buf211  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, reinterpret_tensor(buf325, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_167, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf327)
    del primals_168
    buf328 = buf203; del buf203  # reuse
    # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, reinterpret_tensor(buf325, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_169, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf328)
    del primals_170
    buf329 = reinterpret_tensor(buf326, (8, 1, 128), (128, 128, 1), 0); del buf326  # reuse
    buf330 = empty((8, 4, 32, 401), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_43(c_void_p(buf329.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf330.data_ptr()))
    del primals_166
    buf331 = empty((32, 1, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf330, (32, 32, 401), (12832, 401, 1), 0), out=buf331)
    buf332 = buf306; del buf306  # reuse
    buf333 = reinterpret_tensor(buf331, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf331  # reuse
    buf334 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf335 = empty((8, 4, 1, 401), device='cpu', dtype=torch.float32)
    buf336 = reinterpret_tensor(buf327, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf327  # reuse
    cpp_fused__softmax_clone_mul_44(c_void_p(buf333.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf293, (32, 1, 32), (32, 32, 1), 0); del buf293  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf336, (32, 401, 32), (12832, 32, 1), 0), out=buf337)
    buf338 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, reinterpret_tensor(buf337, (8, 128), (128, 1), 0), reinterpret_tensor(primals_171, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf338)
    del primals_172
    buf339 = buf313; del buf313  # reuse
    buf340 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf342 = empty((8, 1, 128), device='cpu', dtype=torch.float32)
    buf343 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_45(c_void_p(buf320.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    buf344 = buf312; del buf312  # reuse
    # Source Nodes: [reverted_proj_cls_token_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf343, reinterpret_tensor(primals_175, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf344)
    del primals_176
    buf345 = reinterpret_tensor(buf302, (8, 197, 256), (50432, 256, 1), 0); del buf302  # reuse
    buf346 = empty((8, 401, 1), device='cpu', dtype=torch.float32)
    buf347 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf349 = reinterpret_tensor(buf347, (8, 401, 1), (401, 1, 1), 0); del buf347  # reuse
    buf350 = buf328; del buf328  # reuse
    cpp_fused_cat_native_layer_norm_view_46(c_void_p(buf349.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf350.data_ptr()))
    del primals_178
    buf351 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf350, reinterpret_tensor(primals_179, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf351)
    del primals_180
    # Source Nodes: [x_115], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf352 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 256))
    buf353 = buf352[0]
    buf354 = buf352[1]
    buf355 = buf352[2]
    buf356 = buf352[3]
    buf357 = buf352[6]
    buf358 = buf352[7]
    del buf352
    buf360 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, reinterpret_tensor(buf353, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_181, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf360)
    del primals_182
    buf361 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf362 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf364 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf365 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_47(c_void_p(buf319.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del primals_184
    buf366 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_186, buf365, reinterpret_tensor(primals_185, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf366)
    del primals_186
    buf367 = empty((3208, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_48(c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    buf368 = empty((3208, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf367, reinterpret_tensor(primals_187, (384, 128), (1, 384), 0), alpha=1, beta=1, out=buf368)
    del primals_188
    buf369 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf370, (8, 197, 1), (197, 1, 1), 0); del buf370  # reuse
    buf373 = buf281; del buf281  # reuse
    cpp_fused_native_layer_norm_view_49(c_void_p(buf372.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf373.data_ptr()))
    del primals_190
    buf374 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_192, buf373, reinterpret_tensor(primals_191, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf374)
    del primals_192
    # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf375 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf376 = buf375[0]
    buf377 = buf375[1]
    buf378 = buf375[2]
    buf379 = buf375[3]
    buf380 = buf375[6]
    buf381 = buf375[7]
    del buf375
    buf383 = buf273; del buf273  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, reinterpret_tensor(buf376, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_193, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf383)
    del primals_194
    buf384 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf385 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf387 = buf258; del buf258  # reuse
    buf388 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_50(c_void_p(buf345.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del primals_196
    buf389 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_198, buf388, reinterpret_tensor(primals_197, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf389)
    del primals_198
    buf390 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_51(c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf390, reinterpret_tensor(primals_199, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf391)
    del primals_200
    buf392 = buf384; del buf384  # reuse
    buf393 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf395 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf396 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_52(c_void_p(buf345.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del primals_202
    buf397 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf396, reinterpret_tensor(primals_203, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf397)
    del primals_204
    # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf398 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf399 = buf398[0]
    buf400 = buf398[1]
    buf401 = buf398[2]
    buf402 = buf398[3]
    buf403 = buf398[6]
    buf404 = buf398[7]
    del buf398
    buf406 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_206, reinterpret_tensor(buf399, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_205, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf406)
    del primals_206
    buf407 = buf392; del buf392  # reuse
    buf408 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf410 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf411 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf345.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del primals_208
    buf412 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, buf411, reinterpret_tensor(primals_209, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf412)
    del primals_210
    buf413 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_54(c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    buf414 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_212, buf413, reinterpret_tensor(primals_211, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf414)
    del primals_212
    buf415 = reinterpret_tensor(buf414, (8, 197, 256), (50432, 256, 1), 0); del buf414  # reuse
    buf416 = buf407; del buf407  # reuse
    buf417 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf419 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf420 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf415.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()))
    del primals_214
    buf421 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_216, buf420, reinterpret_tensor(primals_215, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf421)
    del primals_216
    # Source Nodes: [x_151], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf422 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 512))
    buf423 = buf422[0]
    buf424 = buf422[1]
    buf425 = buf422[2]
    buf426 = buf422[3]
    buf427 = buf422[6]
    buf428 = buf422[7]
    del buf422
    buf430 = buf406; del buf406  # reuse
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_218, reinterpret_tensor(buf423, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_217, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf430)
    del primals_218
    buf431 = buf416; del buf416  # reuse
    buf432 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf434 = reinterpret_tensor(buf391, (8, 197, 256), (50432, 256, 1), 0); del buf391  # reuse
    buf435 = buf383; del buf383  # reuse
    cpp_fused_add_native_layer_norm_view_56(c_void_p(buf415.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()))
    del primals_220
    buf436 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_222, buf435, reinterpret_tensor(primals_221, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf436)
    del primals_222
    buf437 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_57(c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    buf438 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_224, buf437, reinterpret_tensor(primals_223, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf438)
    del primals_224
    buf439 = buf339; del buf339  # reuse
    buf440 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf442 = reinterpret_tensor(buf338, (8, 1, 128), (128, 128, 1), 0); del buf338  # reuse
    buf443 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_58(c_void_p(buf319.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()))
    buf444 = buf344; del buf344  # reuse
    # Source Nodes: [l__mod___blocks_2_projs_0_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_228, buf443, reinterpret_tensor(primals_227, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf444)
    del primals_228
    buf445 = buf439; del buf439  # reuse
    buf446 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf448 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf449 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_59(c_void_p(buf415.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    buf450 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_projs_1_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_232, buf449, reinterpret_tensor(primals_231, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf450)
    del primals_232
    buf451 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf452 = reinterpret_tensor(buf431, (8, 197, 1), (197, 1, 1), 0); del buf431  # reuse
    buf453 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf455 = reinterpret_tensor(buf453, (8, 197, 1), (197, 1, 1), 0); del buf453  # reuse
    buf456 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_60(c_void_p(buf455.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf456.data_ptr()))
    del primals_234
    buf457 = buf444; del buf444  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf456, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_235, (256, 256), (1, 256), 0), out=buf457)
    buf458 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_238, reinterpret_tensor(buf456, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_237, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf458)
    del primals_238
    buf459 = empty((1576, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_240, reinterpret_tensor(buf456, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_239, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf459)
    del primals_240
    buf460 = reinterpret_tensor(buf457, (8, 1, 256), (256, 256, 1), 0); del buf457  # reuse
    buf461 = empty((8, 4, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_61(c_void_p(buf460.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf461.data_ptr()))
    del primals_236
    buf462 = empty((32, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf460, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf461, (32, 64, 197), (12608, 197, 1), 0), out=buf462)
    buf463 = buf332; del buf332  # reuse
    buf464 = reinterpret_tensor(buf462, (8, 4, 1, 197), (788, 197, 6304, 1), 0); del buf462  # reuse
    buf465 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf466 = empty((8, 4, 1, 197), device='cpu', dtype=torch.float32)
    buf467 = reinterpret_tensor(buf458, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf458  # reuse
    cpp_fused__softmax_clone_mul_62(c_void_p(buf464.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()))
    buf468 = empty((32, 1, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf466, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf467, (32, 197, 64), (12608, 64, 1), 0), out=buf468)
    buf469 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_242, reinterpret_tensor(buf468, (8, 256), (256, 1), 0), reinterpret_tensor(primals_241, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf469)
    del primals_242
    buf470 = buf445; del buf445  # reuse
    buf471 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf473 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    buf474 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_63(c_void_p(buf451.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()))
    buf475 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [reverted_proj_cls_token_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_246, buf474, reinterpret_tensor(primals_245, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf475)
    del primals_246
    buf476 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf477 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf478 = reinterpret_tensor(buf361, (8, 401, 1), (401, 1, 1), 0); del buf361  # reuse
    buf479 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf481 = reinterpret_tensor(buf479, (8, 401, 1), (401, 1, 1), 0); del buf479  # reuse
    buf482 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_64(c_void_p(buf481.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf482.data_ptr()))
    del primals_248
    buf483 = buf475; del buf475  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_249, (128, 128), (1, 128), 0), out=buf483)
    buf484 = buf368; del buf368  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wk], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_252, reinterpret_tensor(buf482, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_251, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf484)
    del primals_252
    buf485 = buf360; del buf360  # reuse
    # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_254, reinterpret_tensor(buf482, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_253, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf485)
    del primals_254
    buf486 = reinterpret_tensor(buf483, (8, 1, 128), (128, 128, 1), 0); del buf483  # reuse
    buf487 = empty((8, 4, 32, 401), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_65(c_void_p(buf486.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf487.data_ptr()))
    del primals_250
    buf488 = empty((32, 1, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf486, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf487, (32, 32, 401), (12832, 401, 1), 0), out=buf488)
    buf489 = buf463; del buf463  # reuse
    buf490 = reinterpret_tensor(buf488, (8, 4, 1, 401), (1604, 401, 12832, 1), 0); del buf488  # reuse
    buf491 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf492 = empty((8, 4, 1, 401), device='cpu', dtype=torch.float32)
    buf493 = reinterpret_tensor(buf484, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf484  # reuse
    cpp_fused__softmax_clone_mul_66(c_void_p(buf490.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()))
    del buf489
    buf494 = reinterpret_tensor(buf450, (32, 1, 32), (32, 32, 1), 0); del buf450  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf492, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf493, (32, 401, 32), (12832, 32, 1), 0), out=buf494)
    buf495 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, reinterpret_tensor(buf494, (8, 128), (128, 1), 0), reinterpret_tensor(primals_255, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf495)
    del primals_256
    buf496 = buf470; del buf470  # reuse
    buf497 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf499 = empty((8, 1, 128), device='cpu', dtype=torch.float32)
    buf500 = empty((8, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_native_layer_norm_view_67(c_void_p(buf477.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del buf496
    buf501 = buf469; del buf469  # reuse
    # Source Nodes: [reverted_proj_cls_token_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_260, buf500, reinterpret_tensor(primals_259, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf501)
    del primals_260
    buf502 = reinterpret_tensor(buf459, (8, 197, 256), (50432, 256, 1), 0); del buf459  # reuse
    buf503 = empty((8, 401, 1), device='cpu', dtype=torch.float32)
    buf504 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf506 = reinterpret_tensor(buf504, (8, 401, 1), (401, 1, 1), 0); del buf504  # reuse
    buf507 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf508 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf510 = reinterpret_tensor(buf508, (8, 197, 1), (197, 1, 1), 0); del buf508  # reuse
    buf511 = buf495; del buf495  # reuse
    buf512 = empty((8, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_clone_native_layer_norm_68(c_void_p(buf506.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()))
    del buf501
    del primals_262
    del primals_264
    buf513 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_266, buf511, reinterpret_tensor(primals_265, (128, 1000), (1, 128), 0), alpha=1, beta=1, out=buf513)
    del primals_266
    buf514 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_268, buf512, reinterpret_tensor(primals_267, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf514)
    del primals_268
    buf515 = empty((8, 1000), device='cpu', dtype=torch.float32)
    buf516 = reinterpret_tensor(buf497, (8, 1, 1), (1, 1, 1), 0); del buf497  # reuse
    buf517 = empty_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    buf518 = reinterpret_tensor(buf471, (8, 1, 1), (1, 1, 1), 0); del buf471  # reuse
    buf519 = empty_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    buf520 = reinterpret_tensor(buf446, (8, 1, 1), (1, 1, 1), 0); del buf446  # reuse
    buf521 = reinterpret_tensor(buf440, (8, 1, 1), (1, 1, 1), 0); del buf440  # reuse
    buf522 = reinterpret_tensor(buf432, (8, 197, 1), (197, 1, 1), 0); del buf432  # reuse
    buf523 = reinterpret_tensor(buf438, (8, 4, 197, 64), (50432, 1, 256, 4), 0); del buf438  # reuse
    buf524 = reinterpret_tensor(buf417, (8, 197, 1), (197, 1, 1), 0); del buf417  # reuse
    buf525 = reinterpret_tensor(buf408, (8, 197, 1), (197, 1, 1), 0); del buf408  # reuse
    buf526 = reinterpret_tensor(buf430, (8, 4, 197, 64), (50432, 1, 256, 4), 0); del buf430  # reuse
    buf527 = reinterpret_tensor(buf393, (8, 197, 1), (197, 1, 1), 0); del buf393  # reuse
    buf528 = reinterpret_tensor(buf385, (8, 197, 1), (197, 1, 1), 0); del buf385  # reuse
    buf529 = reinterpret_tensor(buf415, (8, 4, 197, 64), (50432, 1, 256, 4), 0); del buf415  # reuse
    buf530 = reinterpret_tensor(buf362, (8, 401, 1), (401, 1, 1), 0); del buf362  # reuse
    buf531 = reinterpret_tensor(buf485, (8, 4, 401, 32), (51328, 1, 128, 4), 0); del buf485  # reuse
    buf532 = reinterpret_tensor(buf340, (8, 1, 1), (1, 1, 1), 0); del buf340  # reuse
    buf533 = empty_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    buf534 = reinterpret_tensor(buf314, (8, 1, 1), (1, 1, 1), 0); del buf314  # reuse
    buf535 = empty_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    buf536 = reinterpret_tensor(buf289, (8, 1, 1), (1, 1, 1), 0); del buf289  # reuse
    buf537 = reinterpret_tensor(buf283, (8, 1, 1), (1, 1, 1), 0); del buf283  # reuse
    buf538 = reinterpret_tensor(buf275, (8, 197, 1), (197, 1, 1), 0); del buf275  # reuse
    buf539 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf540 = reinterpret_tensor(buf260, (8, 197, 1), (197, 1, 1), 0); del buf260  # reuse
    buf541 = reinterpret_tensor(buf251, (8, 197, 1), (197, 1, 1), 0); del buf251  # reuse
    buf542 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf543 = reinterpret_tensor(buf236, (8, 197, 1), (197, 1, 1), 0); del buf236  # reuse
    buf544 = reinterpret_tensor(buf228, (8, 197, 1), (197, 1, 1), 0); del buf228  # reuse
    buf545 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf546 = reinterpret_tensor(buf205, (8, 401, 1), (401, 1, 1), 0); del buf205  # reuse
    buf547 = empty_strided((8, 4, 401, 32), (51328, 1, 128, 4), device='cpu', dtype=torch.float32)
    buf548 = reinterpret_tensor(buf183, (8, 1, 1), (1, 1, 1), 0); del buf183  # reuse
    buf549 = empty_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    buf550 = reinterpret_tensor(buf157, (8, 1, 1), (1, 1, 1), 0); del buf157  # reuse
    buf551 = empty_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    buf552 = reinterpret_tensor(buf132, (8, 1, 1), (1, 1, 1), 0); del buf132  # reuse
    buf553 = reinterpret_tensor(buf126, (8, 1, 1), (1, 1, 1), 0); del buf126  # reuse
    buf554 = reinterpret_tensor(buf117, (8, 197, 1), (197, 1, 1), 0); del buf117  # reuse
    buf555 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf556 = reinterpret_tensor(buf102, (8, 197, 1), (197, 1, 1), 0); del buf102  # reuse
    buf557 = reinterpret_tensor(buf94, (8, 197, 1), (197, 1, 1), 0); del buf94  # reuse
    buf558 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf559 = reinterpret_tensor(buf79, (8, 197, 1), (197, 1, 1), 0); del buf79  # reuse
    buf560 = reinterpret_tensor(buf70, (8, 197, 1), (197, 1, 1), 0); del buf70  # reuse
    buf561 = empty_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    buf562 = reinterpret_tensor(buf55, (8, 197, 1), (197, 1, 1), 0); del buf55  # reuse
    buf563 = reinterpret_tensor(buf46, (8, 401, 1), (401, 1, 1), 0); del buf46  # reuse
    buf564 = empty_strided((8, 4, 401, 32), (51328, 1, 128, 4), device='cpu', dtype=torch.float32)
    buf565 = reinterpret_tensor(buf31, (8, 401, 1), (401, 1, 1), 0); del buf31  # reuse
    cpp_fused__softmax_add_cat_detach_mean_native_layer_norm_native_layer_norm_backward_69(c_void_p(buf516.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf564.data_ptr()))
    return (buf515, buf0, buf1, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, buf2, buf28, buf33, buf34, reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf35, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf38, buf39, buf40, buf41, buf42, reinterpret_tensor(buf37, (3208, 128), (128, 1), 0), buf48, buf49, buf50, buf51, buf57, buf58, reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf59, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf62, buf63, buf64, buf65, buf66, reinterpret_tensor(buf61, (1576, 256), (256, 1), 0), buf72, buf73, buf74, buf75, buf81, buf82, reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf83, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf86, buf87, buf88, buf89, buf90, reinterpret_tensor(buf85, (1576, 256), (256, 1), 0), buf96, buf97, buf98, buf99, buf104, buf105, reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf106, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf109, buf110, buf111, buf112, buf113, reinterpret_tensor(buf108, (1576, 256), (256, 1), 0), buf119, buf120, buf121, buf122, buf128, buf129, buf134, buf135, buf137, buf138, buf141, reinterpret_tensor(buf142, (8, 256), (50432, 1), 0), reinterpret_tensor(buf142, (1576, 256), (256, 1), 0), reinterpret_tensor(buf154, (8, 256), (256, 1), 0), buf159, buf160, buf162, buf163, buf164, buf167, reinterpret_tensor(buf168, (8, 128), (51328, 1), 0), reinterpret_tensor(buf168, (3208, 128), (128, 1), 0), reinterpret_tensor(buf180, (8, 128), (128, 1), 0), buf185, buf186, buf188, buf189, buf192, buf193, reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf194, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf197, buf198, buf199, buf200, buf201, reinterpret_tensor(buf196, (3208, 128), (128, 1), 0), buf207, buf208, buf209, buf210, buf212, buf215, buf216, reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf217, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf220, buf221, buf222, buf223, buf224, reinterpret_tensor(buf219, (1576, 256), (256, 1), 0), buf230, buf231, buf232, buf233, buf238, buf239, reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf240, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf243, buf244, buf245, buf246, buf247, reinterpret_tensor(buf242, (1576, 256), (256, 1), 0), buf253, buf254, buf255, buf256, buf262, buf263, reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf264, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf267, buf268, buf269, buf270, buf271, reinterpret_tensor(buf266, (1576, 256), (256, 1), 0), buf277, buf278, buf279, buf280, buf285, buf286, buf291, buf292, buf294, buf295, buf298, reinterpret_tensor(buf299, (8, 256), (50432, 1), 0), reinterpret_tensor(buf299, (1576, 256), (256, 1), 0), reinterpret_tensor(buf311, (8, 256), (256, 1), 0), buf316, buf317, buf319, buf320, buf321, buf324, reinterpret_tensor(buf325, (8, 128), (51328, 1), 0), reinterpret_tensor(buf325, (3208, 128), (128, 1), 0), reinterpret_tensor(buf337, (8, 128), (128, 1), 0), buf342, buf343, buf345, buf346, buf349, buf350, reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf351, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf354, buf355, buf356, buf357, buf358, reinterpret_tensor(buf353, (3208, 128), (128, 1), 0), buf364, buf365, buf366, buf367, buf369, buf372, buf373, reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf374, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf377, buf378, buf379, buf380, buf381, reinterpret_tensor(buf376, (1576, 256), (256, 1), 0), buf387, buf388, buf389, buf390, buf395, buf396, reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf397, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf400, buf401, buf402, buf403, buf404, reinterpret_tensor(buf399, (1576, 256), (256, 1), 0), buf410, buf411, buf412, buf413, buf419, buf420, reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf421, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf424, buf425, buf426, buf427, buf428, reinterpret_tensor(buf423, (1576, 256), (256, 1), 0), buf434, buf435, buf436, buf437, buf442, buf443, buf448, buf449, buf451, buf452, buf455, reinterpret_tensor(buf456, (8, 256), (50432, 1), 0), reinterpret_tensor(buf456, (1576, 256), (256, 1), 0), reinterpret_tensor(buf468, (8, 256), (256, 1), 0), buf473, buf474, buf476, buf477, buf478, buf481, reinterpret_tensor(buf482, (8, 128), (51328, 1), 0), reinterpret_tensor(buf482, (3208, 128), (128, 1), 0), reinterpret_tensor(buf494, (8, 128), (128, 1), 0), buf499, buf500, buf502, buf503, buf506, buf507, buf510, buf511, buf512, reinterpret_tensor(primals_267, (1000, 256), (256, 1), 0), reinterpret_tensor(primals_265, (1000, 128), (128, 1), 0), reinterpret_tensor(primals_259, (256, 128), (128, 1), 0), buf516, reinterpret_tensor(primals_255, (128, 128), (128, 1), 0), reinterpret_tensor(buf492, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf493, (32, 32, 401), (12832, 1, 32), 0), buf517, reinterpret_tensor(buf486, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf487, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_253, (128, 128), (128, 1), 0), reinterpret_tensor(primals_251, (128, 128), (128, 1), 0), reinterpret_tensor(primals_249, (128, 128), (128, 1), 0), reinterpret_tensor(primals_245, (128, 256), (256, 1), 0), buf518, reinterpret_tensor(primals_241, (256, 256), (256, 1), 0), reinterpret_tensor(buf466, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf467, (32, 64, 197), (12608, 1, 64), 0), buf519, reinterpret_tensor(buf460, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf461, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_239, (256, 256), (256, 1), 0), reinterpret_tensor(primals_237, (256, 256), (256, 1), 0), reinterpret_tensor(primals_235, (256, 256), (256, 1), 0), reinterpret_tensor(primals_231, (128, 256), (256, 1), 0), buf520, reinterpret_tensor(primals_227, (256, 128), (128, 1), 0), buf521, reinterpret_tensor(primals_223, (256, 768), (768, 1), 0), reinterpret_tensor(primals_221, (768, 256), (256, 1), 0), buf522, reinterpret_tensor(primals_217, (256, 256), (256, 1), 0), buf523, reinterpret_tensor(primals_215, (768, 256), (256, 1), 0), buf524, reinterpret_tensor(primals_211, (256, 768), (768, 1), 0), reinterpret_tensor(primals_209, (768, 256), (256, 1), 0), buf525, reinterpret_tensor(primals_205, (256, 256), (256, 1), 0), buf526, reinterpret_tensor(primals_203, (768, 256), (256, 1), 0), buf527, reinterpret_tensor(primals_199, (256, 768), (768, 1), 0), reinterpret_tensor(primals_197, (768, 256), (256, 1), 0), buf528, reinterpret_tensor(primals_193, (256, 256), (256, 1), 0), buf529, reinterpret_tensor(primals_191, (768, 256), (256, 1), 0), reinterpret_tensor(primals_187, (128, 384), (384, 1), 0), reinterpret_tensor(primals_185, (384, 128), (128, 1), 0), buf530, reinterpret_tensor(primals_181, (128, 128), (128, 1), 0), buf531, reinterpret_tensor(primals_179, (384, 128), (128, 1), 0), reinterpret_tensor(primals_175, (256, 128), (128, 1), 0), buf532, reinterpret_tensor(primals_171, (128, 128), (128, 1), 0), reinterpret_tensor(buf335, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf336, (32, 32, 401), (12832, 1, 32), 0), buf533, reinterpret_tensor(buf329, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf330, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_169, (128, 128), (128, 1), 0), reinterpret_tensor(primals_167, (128, 128), (128, 1), 0), reinterpret_tensor(primals_165, (128, 128), (128, 1), 0), reinterpret_tensor(primals_161, (128, 256), (256, 1), 0), buf534, reinterpret_tensor(primals_157, (256, 256), (256, 1), 0), reinterpret_tensor(buf309, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf310, (32, 64, 197), (12608, 1, 64), 0), buf535, reinterpret_tensor(buf303, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf304, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_155, (256, 256), (256, 1), 0), reinterpret_tensor(primals_153, (256, 256), (256, 1), 0), reinterpret_tensor(primals_151, (256, 256), (256, 1), 0), reinterpret_tensor(primals_147, (128, 256), (256, 1), 0), buf536, reinterpret_tensor(primals_143, (256, 128), (128, 1), 0), buf537, reinterpret_tensor(primals_139, (256, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 256), (256, 1), 0), buf538, reinterpret_tensor(primals_133, (256, 256), (256, 1), 0), buf539, reinterpret_tensor(primals_131, (768, 256), (256, 1), 0), buf540, reinterpret_tensor(primals_127, (256, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 256), (256, 1), 0), buf541, reinterpret_tensor(primals_121, (256, 256), (256, 1), 0), buf542, reinterpret_tensor(primals_119, (768, 256), (256, 1), 0), buf543, reinterpret_tensor(primals_115, (256, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 256), (256, 1), 0), buf544, reinterpret_tensor(primals_109, (256, 256), (256, 1), 0), buf545, reinterpret_tensor(primals_107, (768, 256), (256, 1), 0), reinterpret_tensor(primals_103, (128, 384), (384, 1), 0), reinterpret_tensor(primals_101, (384, 128), (128, 1), 0), buf546, reinterpret_tensor(primals_97, (128, 128), (128, 1), 0), buf547, reinterpret_tensor(primals_95, (384, 128), (128, 1), 0), reinterpret_tensor(primals_91, (256, 128), (128, 1), 0), buf548, reinterpret_tensor(primals_87, (128, 128), (128, 1), 0), reinterpret_tensor(buf178, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf179, (32, 32, 401), (12832, 1, 32), 0), buf549, reinterpret_tensor(buf172, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf173, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_85, (128, 128), (128, 1), 0), reinterpret_tensor(primals_83, (128, 128), (128, 1), 0), reinterpret_tensor(primals_81, (128, 128), (128, 1), 0), reinterpret_tensor(primals_77, (128, 256), (256, 1), 0), buf550, reinterpret_tensor(primals_73, (256, 256), (256, 1), 0), reinterpret_tensor(buf152, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf153, (32, 64, 197), (12608, 1, 64), 0), buf551, reinterpret_tensor(buf146, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf147, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_71, (256, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 256), (256, 1), 0), reinterpret_tensor(primals_67, (256, 256), (256, 1), 0), reinterpret_tensor(primals_63, (128, 256), (256, 1), 0), buf552, reinterpret_tensor(primals_59, (256, 128), (128, 1), 0), buf553, reinterpret_tensor(primals_55, (256, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 256), (256, 1), 0), buf554, reinterpret_tensor(primals_49, (256, 256), (256, 1), 0), buf555, reinterpret_tensor(primals_47, (768, 256), (256, 1), 0), buf556, reinterpret_tensor(primals_43, (256, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 256), (256, 1), 0), buf557, reinterpret_tensor(primals_37, (256, 256), (256, 1), 0), buf558, reinterpret_tensor(primals_35, (768, 256), (256, 1), 0), buf559, reinterpret_tensor(primals_31, (256, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 256), (256, 1), 0), buf560, reinterpret_tensor(primals_25, (256, 256), (256, 1), 0), buf561, reinterpret_tensor(primals_23, (768, 256), (256, 1), 0), buf562, reinterpret_tensor(primals_19, (128, 384), (384, 1), 0), reinterpret_tensor(primals_17, (384, 128), (128, 1), 0), buf563, reinterpret_tensor(primals_13, (128, 128), (128, 1), 0), buf564, reinterpret_tensor(primals_11, (384, 128), (128, 1), 0), buf565, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1000, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((1000, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
