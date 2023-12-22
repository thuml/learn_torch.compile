
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(24L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-2712L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>((-2688L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>((-2664L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>((-24L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(24L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(2664L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(2688L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(2712L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>((-2712L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>((-2688L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>((-112L) + (2L*x2) + (224L*x1));
                            auto tmp78 = c10::convert<long>((-113L) + (2L*x2) + (224L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>((-2664L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>((-111L) + (2L*x2) + (224L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>((-24L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>((-1L) + (2L*x2) + (224L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>((2L*x2) + (224L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(24L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(1L + (2L*x2) + (224L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(2664L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(111L + (2L*x2) + (224L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(2688L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(112L + (2L*x2) + (224L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(2712L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(113L + (2L*x2) + (224L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (24L*x2) + (1344L*x1) + (75264L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (24L*x2) + (1344L*x1) + (75264L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x2) + (45472L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1)];
                        auto tmp3 = in_ptr2[static_cast<long>(x1)];
                        auto tmp11 = in_ptr3[static_cast<long>(x1)];
                        auto tmp13 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = std::sqrt(tmp5);
                        auto tmp7 = 1 / tmp6;
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                        auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = tmp14 * (tmp14>0);
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp15 <= tmp16;
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (90944L*x0))] = tmp15;
                        out_ptr1[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x2) + (45472L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1)];
                        auto tmp3 = in_ptr2[static_cast<long>(x1)];
                        auto tmp11 = in_ptr3[static_cast<long>(x1)];
                        auto tmp13 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = std::sqrt(tmp5);
                        auto tmp7 = 1 / tmp6;
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                        auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = tmp14 * (tmp14>0);
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp15 <= tmp16;
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (90944L*x0))] = tmp15;
                        out_ptr1[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp17;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3 + (784L*x1) + (45472L*x2) + (90944L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (784L*x2) + (1568L*x1) + (90944L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(45472L + x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (58L*x2) + (45472L*x0)), static_cast<long>(58L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(45472L + x2 + (784L*x1) + (90944L*x0))];
                        out_ptr3[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (58L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(58);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (784L*x2) + (45472L*x1) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-58L) + x2 + (58L*x1) + (58L*x3) + (45472L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (784L*x1) + (1568L*x2) + (90944L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(45472L + x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (58L*x2) + (45472L*x0)), static_cast<long>(58L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(45472L + x2 + (784L*x1) + (90944L*x0))];
                        out_ptr2[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (58L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(58);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (784L*x2) + (45472L*x1) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-58L) + x2 + (58L*x1) + (58L*x3) + (45472L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (784L*x1) + (1568L*x2) + (90944L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(45472L + x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (58L*x2) + (45472L*x0)), static_cast<long>(58L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(45472L + x2 + (784L*x1) + (90944L*x0))];
                        out_ptr2[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((58L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L)));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(58);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (28L*x1) + (784L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(2L))) + (1568L*(static_cast<long>(c10::div_floor_integer(((58L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L))), 2L)) % static_cast<long>(58L))) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-58L) + (58L*x3) + (58L*(static_cast<long>(x2) % static_cast<long>(2L))) + (1624L*x1) + (45472L*x0) + (c10::div_floor_integer(x2, 2L)))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x2 + (116L*x3) + (3248L*x1) + (90944L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x2 + (196L*x1) + (45472L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp17;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x2 + (196L*x1) + (45472L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp17;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3 + (196L*x1) + (22736L*x2) + (45472L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (392L*x1) + (45472L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x3 + (196L*x1) + (22736L*x2) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (392L*x1) + (45472L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr3 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr3[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2 + (116L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((116L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L)));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(116);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (14L*x1) + (196L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(c10::div_floor_integer(((116L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L))), 2L)) % static_cast<long>(116L))) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-116L) + (116L*x3) + (116L*(static_cast<long>(x2) % static_cast<long>(2L))) + (1624L*x1) + (22736L*x0) + (c10::div_floor_integer(x2, 2L)))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            out_ptr1[static_cast<long>(x2 + (232L*x3) + (3248L*x1) + (45472L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (232L*x2) + (11368L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x2 + (49L*x1) + (22736L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (232L*x2) + (11368L*x0))] = tmp17;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (232L*x2) + (11368L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x2 + (49L*x1) + (22736L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (232L*x2) + (11368L*x0))] = tmp17;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3 + (49L*x1) + (11368L*x2) + (22736L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x3 + (49L*x2) + (98L*x1) + (22736L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x3 + (49L*x1) + (11368L*x2) + (22736L*x0))];
                        out_ptr2[static_cast<long>(x3 + (49L*x2) + (98L*x1) + (22736L*x0))] = tmp0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)), static_cast<long>(232L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2 + (232L*x1));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(232);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr5[static_cast<long>(x3 + (49L*x2) + (11368L*x1) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = out_ptr0[static_cast<long>((-232L) + x2 + (232L*x1) + (232L*x3) + (11368L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        out_ptr1[static_cast<long>(x3 + (49L*x1) + (98L*x2) + (22736L*x0))] = tmp14;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)), static_cast<long>(232L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2 + (232L*x1));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(232);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr5[static_cast<long>(x3 + (49L*x2) + (11368L*x1) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = out_ptr0[static_cast<long>((-232L) + x2 + (232L*x1) + (232L*x3) + (11368L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        out_ptr1[static_cast<long>(x3 + (49L*x1) + (98L*x2) + (22736L*x0))] = tmp14;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)), static_cast<long>(232L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(464L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((232L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L)));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(232);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr5[static_cast<long>(x3 + (7L*x1) + (49L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(2L))) + (98L*(static_cast<long>(c10::div_floor_integer(((232L*(static_cast<long>(x2) % static_cast<long>(2L))) + (c10::div_floor_integer(x2, 2L))), 2L)) % static_cast<long>(232L))) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = out_ptr0[static_cast<long>((-232L) + (232L*x3) + (232L*(static_cast<long>(x2) % static_cast<long>(2L))) + (1624L*x1) + (11368L*x0) + (c10::div_floor_integer(x2, 2L)))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        out_ptr1[static_cast<long>(x2 + (464L*x3) + (3248L*x1) + (22736L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4,
                       bool* out_ptr5,
                       bool* out_ptr6,
                       bool* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9,
                       bool* out_ptr10,
                       bool* out_ptr11,
                       bool* out_ptr12,
                       bool* out_ptr13)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(45472L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(45472L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(45472L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr3[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr4[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr5[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr6[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr7[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr8[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr9[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr9[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(90944L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr10[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    out_ptr10[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(181888L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr11[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr11[static_cast<long>(x0)] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(181888L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr12[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr12[static_cast<long>(x0)] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(181888L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr13[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr13[static_cast<long>(x0)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_8, (58, ), (1, ))
    assert_size_stride(primals_9, (58, ), (1, ))
    assert_size_stride(primals_10, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_11, (58, ), (1, ))
    assert_size_stride(primals_12, (58, ), (1, ))
    assert_size_stride(primals_13, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (58, ), (1, ))
    assert_size_stride(primals_15, (58, ), (1, ))
    assert_size_stride(primals_16, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_17, (58, ), (1, ))
    assert_size_stride(primals_18, (58, ), (1, ))
    assert_size_stride(primals_19, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_20, (58, ), (1, ))
    assert_size_stride(primals_21, (58, ), (1, ))
    assert_size_stride(primals_22, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (58, ), (1, ))
    assert_size_stride(primals_24, (58, ), (1, ))
    assert_size_stride(primals_25, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_26, (58, ), (1, ))
    assert_size_stride(primals_27, (58, ), (1, ))
    assert_size_stride(primals_28, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_29, (58, ), (1, ))
    assert_size_stride(primals_30, (58, ), (1, ))
    assert_size_stride(primals_31, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (58, ), (1, ))
    assert_size_stride(primals_33, (58, ), (1, ))
    assert_size_stride(primals_34, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_35, (58, ), (1, ))
    assert_size_stride(primals_36, (58, ), (1, ))
    assert_size_stride(primals_37, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_38, (58, ), (1, ))
    assert_size_stride(primals_39, (58, ), (1, ))
    assert_size_stride(primals_40, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (58, ), (1, ))
    assert_size_stride(primals_42, (58, ), (1, ))
    assert_size_stride(primals_43, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_44, (58, ), (1, ))
    assert_size_stride(primals_45, (58, ), (1, ))
    assert_size_stride(primals_46, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (116, ), (1, ))
    assert_size_stride(primals_48, (116, ), (1, ))
    assert_size_stride(primals_49, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_50, (116, ), (1, ))
    assert_size_stride(primals_51, (116, ), (1, ))
    assert_size_stride(primals_52, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_53, (116, ), (1, ))
    assert_size_stride(primals_54, (116, ), (1, ))
    assert_size_stride(primals_55, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (116, ), (1, ))
    assert_size_stride(primals_57, (116, ), (1, ))
    assert_size_stride(primals_58, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_59, (116, ), (1, ))
    assert_size_stride(primals_60, (116, ), (1, ))
    assert_size_stride(primals_61, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_62, (116, ), (1, ))
    assert_size_stride(primals_63, (116, ), (1, ))
    assert_size_stride(primals_64, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (116, ), (1, ))
    assert_size_stride(primals_66, (116, ), (1, ))
    assert_size_stride(primals_67, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_68, (116, ), (1, ))
    assert_size_stride(primals_69, (116, ), (1, ))
    assert_size_stride(primals_70, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_71, (116, ), (1, ))
    assert_size_stride(primals_72, (116, ), (1, ))
    assert_size_stride(primals_73, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (116, ), (1, ))
    assert_size_stride(primals_75, (116, ), (1, ))
    assert_size_stride(primals_76, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_77, (116, ), (1, ))
    assert_size_stride(primals_78, (116, ), (1, ))
    assert_size_stride(primals_79, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_80, (116, ), (1, ))
    assert_size_stride(primals_81, (116, ), (1, ))
    assert_size_stride(primals_82, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (116, ), (1, ))
    assert_size_stride(primals_84, (116, ), (1, ))
    assert_size_stride(primals_85, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_86, (116, ), (1, ))
    assert_size_stride(primals_87, (116, ), (1, ))
    assert_size_stride(primals_88, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_89, (116, ), (1, ))
    assert_size_stride(primals_90, (116, ), (1, ))
    assert_size_stride(primals_91, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (116, ), (1, ))
    assert_size_stride(primals_93, (116, ), (1, ))
    assert_size_stride(primals_94, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_95, (116, ), (1, ))
    assert_size_stride(primals_96, (116, ), (1, ))
    assert_size_stride(primals_97, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_98, (116, ), (1, ))
    assert_size_stride(primals_99, (116, ), (1, ))
    assert_size_stride(primals_100, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (116, ), (1, ))
    assert_size_stride(primals_102, (116, ), (1, ))
    assert_size_stride(primals_103, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_104, (116, ), (1, ))
    assert_size_stride(primals_105, (116, ), (1, ))
    assert_size_stride(primals_106, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_107, (116, ), (1, ))
    assert_size_stride(primals_108, (116, ), (1, ))
    assert_size_stride(primals_109, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (116, ), (1, ))
    assert_size_stride(primals_111, (116, ), (1, ))
    assert_size_stride(primals_112, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_113, (116, ), (1, ))
    assert_size_stride(primals_114, (116, ), (1, ))
    assert_size_stride(primals_115, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_116, (116, ), (1, ))
    assert_size_stride(primals_117, (116, ), (1, ))
    assert_size_stride(primals_118, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (116, ), (1, ))
    assert_size_stride(primals_120, (116, ), (1, ))
    assert_size_stride(primals_121, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_122, (116, ), (1, ))
    assert_size_stride(primals_123, (116, ), (1, ))
    assert_size_stride(primals_124, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (232, ), (1, ))
    assert_size_stride(primals_126, (232, ), (1, ))
    assert_size_stride(primals_127, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_128, (232, ), (1, ))
    assert_size_stride(primals_129, (232, ), (1, ))
    assert_size_stride(primals_130, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_131, (232, ), (1, ))
    assert_size_stride(primals_132, (232, ), (1, ))
    assert_size_stride(primals_133, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (232, ), (1, ))
    assert_size_stride(primals_135, (232, ), (1, ))
    assert_size_stride(primals_136, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_137, (232, ), (1, ))
    assert_size_stride(primals_138, (232, ), (1, ))
    assert_size_stride(primals_139, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_140, (232, ), (1, ))
    assert_size_stride(primals_141, (232, ), (1, ))
    assert_size_stride(primals_142, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (232, ), (1, ))
    assert_size_stride(primals_144, (232, ), (1, ))
    assert_size_stride(primals_145, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_146, (232, ), (1, ))
    assert_size_stride(primals_147, (232, ), (1, ))
    assert_size_stride(primals_148, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_149, (232, ), (1, ))
    assert_size_stride(primals_150, (232, ), (1, ))
    assert_size_stride(primals_151, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (232, ), (1, ))
    assert_size_stride(primals_153, (232, ), (1, ))
    assert_size_stride(primals_154, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_155, (232, ), (1, ))
    assert_size_stride(primals_156, (232, ), (1, ))
    assert_size_stride(primals_157, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_158, (232, ), (1, ))
    assert_size_stride(primals_159, (232, ), (1, ))
    assert_size_stride(primals_160, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (232, ), (1, ))
    assert_size_stride(primals_162, (232, ), (1, ))
    assert_size_stride(primals_163, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_164, (232, ), (1, ))
    assert_size_stride(primals_165, (232, ), (1, ))
    assert_size_stride(primals_166, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1000, 1024), (1024, 1))
    assert_size_stride(primals_170, (1000, ), (1, ))
    assert_size_stride(primals_171, (24, ), (1, ))
    assert_size_stride(primals_172, (24, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (58, ), (1, ))
    assert_size_stride(primals_178, (58, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (58, ), (1, ))
    assert_size_stride(primals_181, (58, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (58, ), (1, ))
    assert_size_stride(primals_184, (58, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (58, ), (1, ))
    assert_size_stride(primals_187, (58, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (58, ), (1, ))
    assert_size_stride(primals_190, (58, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (58, ), (1, ))
    assert_size_stride(primals_193, (58, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (58, ), (1, ))
    assert_size_stride(primals_196, (58, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (58, ), (1, ))
    assert_size_stride(primals_199, (58, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (58, ), (1, ))
    assert_size_stride(primals_202, (58, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (58, ), (1, ))
    assert_size_stride(primals_205, (58, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (58, ), (1, ))
    assert_size_stride(primals_208, (58, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (58, ), (1, ))
    assert_size_stride(primals_211, (58, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (58, ), (1, ))
    assert_size_stride(primals_214, (58, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (116, ), (1, ))
    assert_size_stride(primals_217, (116, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (116, ), (1, ))
    assert_size_stride(primals_220, (116, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (116, ), (1, ))
    assert_size_stride(primals_223, (116, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (116, ), (1, ))
    assert_size_stride(primals_226, (116, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (116, ), (1, ))
    assert_size_stride(primals_229, (116, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (116, ), (1, ))
    assert_size_stride(primals_232, (116, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (116, ), (1, ))
    assert_size_stride(primals_235, (116, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (116, ), (1, ))
    assert_size_stride(primals_238, (116, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (116, ), (1, ))
    assert_size_stride(primals_241, (116, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (116, ), (1, ))
    assert_size_stride(primals_244, (116, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (116, ), (1, ))
    assert_size_stride(primals_247, (116, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (116, ), (1, ))
    assert_size_stride(primals_250, (116, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (116, ), (1, ))
    assert_size_stride(primals_253, (116, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (116, ), (1, ))
    assert_size_stride(primals_256, (116, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (116, ), (1, ))
    assert_size_stride(primals_259, (116, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (116, ), (1, ))
    assert_size_stride(primals_262, (116, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (116, ), (1, ))
    assert_size_stride(primals_265, (116, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (116, ), (1, ))
    assert_size_stride(primals_268, (116, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (116, ), (1, ))
    assert_size_stride(primals_271, (116, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (116, ), (1, ))
    assert_size_stride(primals_274, (116, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (116, ), (1, ))
    assert_size_stride(primals_277, (116, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (116, ), (1, ))
    assert_size_stride(primals_280, (116, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (116, ), (1, ))
    assert_size_stride(primals_283, (116, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (116, ), (1, ))
    assert_size_stride(primals_286, (116, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (116, ), (1, ))
    assert_size_stride(primals_289, (116, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (116, ), (1, ))
    assert_size_stride(primals_292, (116, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (232, ), (1, ))
    assert_size_stride(primals_295, (232, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (232, ), (1, ))
    assert_size_stride(primals_298, (232, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (232, ), (1, ))
    assert_size_stride(primals_301, (232, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (232, ), (1, ))
    assert_size_stride(primals_304, (232, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (232, ), (1, ))
    assert_size_stride(primals_307, (232, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (232, ), (1, ))
    assert_size_stride(primals_310, (232, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (232, ), (1, ))
    assert_size_stride(primals_313, (232, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (232, ), (1, ))
    assert_size_stride(primals_316, (232, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (232, ), (1, ))
    assert_size_stride(primals_319, (232, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (232, ), (1, ))
    assert_size_stride(primals_322, (232, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (232, ), (1, ))
    assert_size_stride(primals_325, (232, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (232, ), (1, ))
    assert_size_stride(primals_328, (232, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (232, ), (1, ))
    assert_size_stride(primals_331, (232, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (232, ), (1, ))
    assert_size_stride(primals_334, (232, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_1
    del primals_339
    # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 24, 112, 112), (301056, 1, 2688, 24))
    buf3 = empty_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(c_void_p(buf2.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_3
    # Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
    buf6 = extern_kernels.convolution(buf4, primals_4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf6, (4, 24, 28, 28), (18816, 1, 672, 24))
    buf7 = empty_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_2(c_void_p(buf6.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_6
    # Source Nodes: [getattr_l__mod___stage2___0___branch1_2], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf16 = empty((4, 116, 28, 28), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf16, (4, 58, 28, 28), (90944, 784, 28, 1), 0)  # alias
    buf170 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_3(c_void_p(buf8.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf170.data_ptr()))
    del primals_9
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf4, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (4, 58, 56, 56), (181888, 1, 3248, 58))
    buf11 = empty_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_4(c_void_p(buf10.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_12
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, primals_13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf12, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf13 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_5(c_void_p(buf12.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_15
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_5], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf15 = reinterpret_tensor(buf16, (4, 58, 28, 28), (90944, 784, 28, 1), 45472)  # alias
    buf169 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    buf17 = empty((4, 58, 2, 28, 28), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_6(c_void_p(buf14.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del buf15
    del buf9
    del primals_18
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf18, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf20 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_7(c_void_p(buf19.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_21
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_3], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf21, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf22 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_8(c_void_p(buf21.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf22.data_ptr()))
    del primals_24
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_5], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(buf22, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf24 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    buf25 = reinterpret_tensor(buf16, (4, 58, 2, 28, 28), (90944, 1568, 784, 28, 1), 0); del buf16  # reuse
    buf26 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_9(c_void_p(buf23.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_27
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf26, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf28 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_10(c_void_p(buf27.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf28.data_ptr()))
    del primals_30
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_3], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf29, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf30 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_11(c_void_p(buf29.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf30.data_ptr()))
    del primals_33
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_5], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf32 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    buf33 = empty((4, 58, 2, 28, 28), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_12(c_void_p(buf31.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del primals_36
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
    buf35 = extern_kernels.convolution(buf34, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf35, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf36 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_13(c_void_p(buf35.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf36.data_ptr()))
    del primals_39
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_3], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf36, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf37, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf38 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_14(c_void_p(buf37.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_42
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_5], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (4, 58, 28, 28), (45472, 1, 1624, 58))
    buf40 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_15(c_void_p(buf39.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_45
    # Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, primals_46, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf42, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf43 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_16(c_void_p(buf42.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_48
    # Source Nodes: [getattr_l__mod___stage3___0___branch1_2], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf52 = empty((4, 232, 14, 14), device='cpu', dtype=torch.float32)
    buf45 = reinterpret_tensor(buf52, (4, 116, 14, 14), (45472, 196, 14, 1), 0)  # alias
    buf165 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_17(c_void_p(buf44.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_51
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf41, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (4, 116, 28, 28), (90944, 1, 3248, 116))
    buf47 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf46.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_54
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_3], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, primals_55, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf48, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf49 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_19(c_void_p(buf48.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_57
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_5], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf51 = reinterpret_tensor(buf52, (4, 116, 14, 14), (45472, 196, 14, 1), 22736)  # alias
    buf164 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf53 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_20(c_void_p(buf50.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf45
    del buf51
    del primals_60
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf55, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf56 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf55.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_63
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_3], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf57, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf58 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_22(c_void_p(buf57.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf58.data_ptr()))
    del primals_66
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_5], Original ATen: [aten.convolution]
    buf59 = extern_kernels.convolution(buf58, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf59, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf60 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf61 = reinterpret_tensor(buf52, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf52  # reuse
    buf62 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_23(c_void_p(buf59.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_69
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf64 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf63.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf64.data_ptr()))
    del primals_72
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_3], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf64, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf65, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf66 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_25(c_void_p(buf65.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf66.data_ptr()))
    del primals_75
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_5], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf68 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf69 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_26(c_void_p(buf67.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_78
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
    buf71 = extern_kernels.convolution(buf70, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf72 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf71.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_81
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_3], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf73, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf74 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_28(c_void_p(buf73.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf74.data_ptr()))
    del primals_84
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_5], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf76 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf77 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_29(c_void_p(buf75.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del primals_87
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
    buf79 = extern_kernels.convolution(buf78, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf80 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf79.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_90
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_3], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf81, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf82 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_31(c_void_p(buf81.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_93
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_5], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(buf82, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf83, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf84 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf85 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf86 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_32(c_void_p(buf83.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del primals_96
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
    buf87 = extern_kernels.convolution(buf86, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf88 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf87.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf88.data_ptr()))
    del primals_99
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_3], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(buf88, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf89, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf90 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_34(c_void_p(buf89.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_102
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_5], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf92 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf93 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_35(c_void_p(buf91.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del primals_105
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf96 = buf94; del buf94  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf95.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_108
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_3], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(buf96, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf97, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf98 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_37(c_void_p(buf97.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf98.data_ptr()))
    del primals_111
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_5], Original ATen: [aten.convolution]
    buf99 = extern_kernels.convolution(buf98, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf100 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf101 = empty((4, 116, 2, 14, 14), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_38(c_void_p(buf99.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_114
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
    buf103 = extern_kernels.convolution(buf102, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf104 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf103.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf104.data_ptr()))
    del primals_117
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_3], Original ATen: [aten.convolution]
    buf105 = extern_kernels.convolution(buf104, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf105, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf106 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_40(c_void_p(buf105.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_120
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_5], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(buf106, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (4, 116, 14, 14), (22736, 1, 1624, 116))
    buf108 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_41(c_void_p(buf107.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_123
    # Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(buf109, primals_124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf110, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf111 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_42(c_void_p(buf110.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf111.data_ptr()))
    del primals_126
    # Source Nodes: [getattr_l__mod___stage4___0___branch1_2], Original ATen: [aten.convolution]
    buf112 = extern_kernels.convolution(buf111, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf112, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf120 = empty((4, 464, 7, 7), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf120, (4, 232, 7, 7), (22736, 49, 7, 1), 0)  # alias
    buf156 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_43(c_void_p(buf112.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf156.data_ptr()))
    del primals_129
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf109, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (4, 232, 14, 14), (45472, 1, 3248, 232))
    buf115 = empty_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf114.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_132
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_3], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, primals_133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf116, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf117 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_45(c_void_p(buf116.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_135
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_5], Original ATen: [aten.convolution]
    buf118 = extern_kernels.convolution(buf117, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf119 = reinterpret_tensor(buf120, (4, 232, 7, 7), (22736, 49, 7, 1), 11368)  # alias
    buf155 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    buf121 = empty((4, 232, 2, 7, 7), device='cpu', dtype=torch.float32)
    buf122 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_threshold_backward_46(c_void_p(buf118.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf113
    del buf119
    del primals_138
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
    buf123 = extern_kernels.convolution(buf122, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf123, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf124 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf123.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf124.data_ptr()))
    del primals_141
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_3], Original ATen: [aten.convolution]
    buf125 = extern_kernels.convolution(buf124, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf125, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf126 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_48(c_void_p(buf125.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_144
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_5], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf126, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf128 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    buf129 = reinterpret_tensor(buf120, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf120  # reuse
    buf130 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_49(c_void_p(buf127.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del primals_147
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf132 = buf130; del buf130  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf131.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_150
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_3], Original ATen: [aten.convolution]
    buf133 = extern_kernels.convolution(buf132, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf133, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf134 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_51(c_void_p(buf133.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf134.data_ptr()))
    del primals_153
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_5], Original ATen: [aten.convolution]
    buf135 = extern_kernels.convolution(buf134, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf135, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf136 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    buf137 = empty((4, 232, 2, 7, 7), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_convolution_relu_52(c_void_p(buf135.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del primals_156
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
    buf139 = extern_kernels.convolution(buf138, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf139, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf140 = buf138; del buf138  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_53(c_void_p(buf139.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf140.data_ptr()))
    del primals_159
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_3], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf140, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf141, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf142 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_54(c_void_p(buf141.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(buf142.data_ptr()))
    del primals_162
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_5], Original ATen: [aten.convolution]
    buf143 = extern_kernels.convolution(buf142, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf143, (4, 232, 7, 7), (11368, 1, 1624, 232))
    buf144 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((4, 464, 7, 7), (22736, 1, 3248, 464), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_relu_view_55(c_void_p(buf143.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_165
    # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf145, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    buf147 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf148 = empty((4, 1024), device='cpu', dtype=torch.float32)
    buf149 = buf148; del buf148  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_56(c_void_p(buf149.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_168
    buf150 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, buf149, reinterpret_tensor(primals_169, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf150)
    del primals_170
    buf151 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    buf152 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    buf153 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    buf154 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    buf157 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf158 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf159 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf160 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf161 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf162 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf163 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    buf166 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    buf167 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    buf168 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    cpp_fused_threshold_backward_57(c_void_p(buf147.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    return (buf150, buf0, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf10, buf11, buf12, buf13, buf14, reinterpret_tensor(buf17, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf19, buf20, buf21, buf22, buf23, reinterpret_tensor(buf25, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf27, buf28, buf29, buf30, buf31, reinterpret_tensor(buf33, (4, 58, 28, 28), (90944, 784, 28, 1), 45472), buf35, buf36, buf37, buf38, buf39, buf41, buf42, buf43, buf44, buf46, buf47, buf48, buf49, buf50, reinterpret_tensor(buf53, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf55, buf56, buf57, buf58, buf59, reinterpret_tensor(buf61, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf63, buf64, buf65, buf66, buf67, reinterpret_tensor(buf69, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf71, buf72, buf73, buf74, buf75, reinterpret_tensor(buf77, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf79, buf80, buf81, buf82, buf83, reinterpret_tensor(buf85, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf87, buf88, buf89, buf90, buf91, reinterpret_tensor(buf93, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf95, buf96, buf97, buf98, buf99, reinterpret_tensor(buf101, (4, 116, 14, 14), (45472, 196, 14, 1), 22736), buf103, buf104, buf105, buf106, buf107, buf109, buf110, buf111, buf112, buf114, buf115, buf116, buf117, buf118, reinterpret_tensor(buf121, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf123, buf124, buf125, buf126, buf127, reinterpret_tensor(buf129, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf131, buf132, buf133, buf134, buf135, reinterpret_tensor(buf137, (4, 232, 7, 7), (22736, 49, 7, 1), 11368), buf139, buf140, buf141, buf142, buf143, buf145, buf146, buf149, reinterpret_tensor(primals_169, (1000, 1024), (1024, 1), 0), buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_174 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_177 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_180 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_183 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_186 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_189 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_195 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_198 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_201 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_204 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_207 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_225 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_228 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_231 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_234 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_237 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_240 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_243 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_246 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_249 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_252 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_255 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_258 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_261 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_264 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_267 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_270 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_273 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_276 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_279 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_282 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_285 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_288 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_294 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_297 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_300 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_303 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_306 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_309 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_312 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_315 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_318 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_321 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_324 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_327 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_330 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_333 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_336 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_339 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
