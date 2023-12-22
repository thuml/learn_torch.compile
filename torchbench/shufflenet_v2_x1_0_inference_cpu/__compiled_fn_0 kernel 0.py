
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


cpp_fused_convolution_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
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
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(24L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-2712L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_out_ptr0 + static_cast<long>((-2688L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = at::vec::maximum(tmp21, tmp13);
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_out_ptr0 + static_cast<long>((-2664L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = at::vec::maximum(tmp30, tmp22);
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_out_ptr0 + static_cast<long>((-24L) + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = at::vec::maximum(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = at::vec::maximum(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_out_ptr0 + static_cast<long>(24L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = at::vec::maximum(tmp49, tmp45);
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_out_ptr0 + static_cast<long>(2664L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = at::vec::maximum(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_out_ptr0 + static_cast<long>(2688L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = at::vec::maximum(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_out_ptr0 + static_cast<long>(2712L + x3 + (48L*x2) + (5376L*x1) + (301056L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = at::vec::maximum(tmp68, tmp64);
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (24L*x2) + (1344L*x1) + (75264L*x0)));
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
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_cat_clone_convolution_5 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(58);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (58L*x1) + (45472L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(1e-05);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(116);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-58L) + x2 + (58L*x1) + (45472L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-58L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-58L) + x2)];
                            auto tmp31 = static_cast<float>(1e-05);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-58L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-58L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (784L*x2) + (90944L*x0))] = tmp44;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (784L*x1) + (45472L*x2) + (90944L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (784L*x2) + (1568L*x1) + (90944L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_clone_convolution_8 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (784L*x2) + (45472L*x1) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-58L) + x2 + (58L*x1) + (58L*x3) + (45472L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (784L*x1) + (1568L*x2) + (90944L*x0))] = tmp29;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(45472L + x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (58L*x2) + (45472L*x0)), static_cast<long>(58L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(45472L + x2 + (784L*x1) + (90944L*x0))];
                        out_ptr1[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_clone_convolution_11 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (784L*x2) + (45472L*x1) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-58L) + x2 + (58L*x1) + (58L*x3) + (45472L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (784L*x1) + (1568L*x2) + (90944L*x0))] = tmp29;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(45472L + x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (58L*x2) + (45472L*x0)), static_cast<long>(58L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(45472L + x2 + (784L*x1) + (90944L*x0))];
                        out_ptr1[static_cast<long>(x1 + (58L*x2) + (45472L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_clone_convolution_14 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (784L*x2) + (45472L*x1) + (90944L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-58L) + x2 + (58L*x1) + (58L*x3) + (45472L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-58L) + x2 + (58L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (784L*x1) + (1568L*x2) + (90944L*x0))] = tmp29;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (90944L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp2 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (90944L*x0)), static_cast<long>(116L));
                        at::vec::transpose_mxn<float,8,8>(tmp2, 8, out_ptr2 + static_cast<long>(x1 + (116L*x2) + (90944L*x0)), static_cast<long>(116L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (784L*x1) + (90944L*x0))];
                        out_ptr1[static_cast<long>(x1 + (116L*x2) + (90944L*x0))] = tmp0;
                        out_ptr2[static_cast<long>(x1 + (116L*x2) + (90944L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1)];
                    auto tmp3 = in_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                    in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_cat_clone_convolution_18 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(116);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (116L*x1) + (22736L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(1e-05);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(232);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1) + (22736L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-116L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-116L) + x2)];
                            auto tmp31 = static_cast<float>(1e-05);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-116L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-116L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (45472L*x0))] = tmp44;
                    }
                }
            }
        }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (196L*x1) + (22736L*x2) + (45472L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (392L*x1) + (45472L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (196L*x1) + (22736L*x2) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (392L*x1) + (45472L*x0))] = tmp0;
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


cpp_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_21 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_24 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_27 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_30 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_33 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_36 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)), static_cast<long>(116L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (116L*x2) + (22736L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(22736L + x2 + (196L*x1) + (45472L*x0))];
                            out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp15;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x1)];
                auto tmp3 = in_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr2[static_cast<long>(x1)];
                auto tmp13 = in_ptr3[static_cast<long>(x1)];
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
                in_out_ptr0[static_cast<long>(x1 + (116L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_clone_convolution_39 = async_compile.cpp('''
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
                                auto tmp6 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (22736L*x1) + (45472L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(232);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-116L) + x2 + (116L*x1) + (116L*x3) + (22736L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                                auto tmp15 = in_ptr3[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp16 = static_cast<float>(1e-05);
                                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                                auto tmp18 = std::sqrt(tmp17);
                                auto tmp19 = 1 / tmp18;
                                auto tmp20 = static_cast<float>(1.0);
                                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                                auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                                auto tmp23 = in_ptr4[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                                auto tmp25 = in_ptr5[static_cast<long>((-116L) + x2 + (116L*x1))];
                                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                                auto tmp27 = tmp26 * (tmp26>0);
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp29 = tmp4 ? tmp7 : tmp28;
                            out_ptr0[static_cast<long>(x3 + (196L*x1) + (392L*x2) + (45472L*x0))] = tmp29;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp2 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (232L*x2) + (45472L*x0)), static_cast<long>(232L));
                        at::vec::transpose_mxn<float,8,8>(tmp2, 8, out_ptr2 + static_cast<long>(x1 + (232L*x2) + (45472L*x0)), static_cast<long>(232L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (45472L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (232L*x2) + (45472L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (232L*x2) + (45472L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_clone_convolution_43 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(464L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(232);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x2 + (232L*x1) + (11368L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x2)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x2)];
                        auto tmp10 = static_cast<float>(1e-05);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x2)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x2)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(464);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr5[static_cast<long>((-232L) + x2 + (232L*x1) + (11368L*x0))];
                        auto tmp28 = in_ptr6[static_cast<long>((-232L) + x2)];
                        auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                        auto tmp30 = in_ptr7[static_cast<long>((-232L) + x2)];
                        auto tmp31 = static_cast<float>(1e-05);
                        auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                        auto tmp33 = std::sqrt(tmp32);
                        auto tmp34 = 1 / tmp33;
                        auto tmp35 = static_cast<float>(1.0);
                        auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                        auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                        auto tmp38 = in_ptr8[static_cast<long>((-232L) + x2)];
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = in_ptr9[static_cast<long>((-232L) + x2)];
                        auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                        auto tmp42 = tmp41 * (tmp41>0);
                        return tmp42;
                    }
                    ;
                    auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp44 = tmp4 ? tmp22 : tmp43;
                    out_ptr0[static_cast<long>(x1 + (49L*x2) + (22736L*x0))] = tmp44;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (49L*x1) + (11368L*x2) + (22736L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (98L*x1) + (22736L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x3 + (49L*x1) + (11368L*x2) + (22736L*x0))];
                        out_ptr1[static_cast<long>(x3 + (49L*x2) + (98L*x1) + (22736L*x0))] = tmp0;
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


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_convolution_46 = async_compile.cpp('''
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
                            auto tmp6 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (11368L*x1) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-232L) + x2 + (232L*x1) + (232L*x3) + (11368L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                            auto tmp15 = in_ptr3[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp16 = static_cast<float>(1e-05);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = std::sqrt(tmp17);
                            auto tmp19 = 1 / tmp18;
                            auto tmp20 = static_cast<float>(1.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                            auto tmp23 = in_ptr4[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = in_ptr5[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                            auto tmp27 = tmp26 * (tmp26>0);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp29 = tmp4 ? tmp7 : tmp28;
                        out_ptr0[static_cast<long>(x3 + (49L*x1) + (98L*x2) + (22736L*x0))] = tmp29;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)), static_cast<long>(232L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_convolution_49 = async_compile.cpp('''
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
                            auto tmp6 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (11368L*x1) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-232L) + x2 + (232L*x1) + (232L*x3) + (11368L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                            auto tmp15 = in_ptr3[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp16 = static_cast<float>(1e-05);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = std::sqrt(tmp17);
                            auto tmp19 = 1 / tmp18;
                            auto tmp20 = static_cast<float>(1.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                            auto tmp23 = in_ptr4[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = in_ptr5[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                            auto tmp27 = tmp26 * (tmp26>0);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp29 = tmp4 ? tmp7 : tmp28;
                        out_ptr0[static_cast<long>(x3 + (49L*x1) + (98L*x2) + (22736L*x0))] = tmp29;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)), static_cast<long>(232L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(11368L + x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (232L*x2) + (11368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_convolution_52 = async_compile.cpp('''
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
                            auto tmp6 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (11368L*x1) + (22736L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(464);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-232L) + x2 + (232L*x1) + (232L*x3) + (11368L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                            auto tmp15 = in_ptr3[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp16 = static_cast<float>(1e-05);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = std::sqrt(tmp17);
                            auto tmp19 = 1 / tmp18;
                            auto tmp20 = static_cast<float>(1.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                            auto tmp23 = in_ptr4[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = in_ptr5[static_cast<long>((-232L) + x2 + (232L*x1))];
                            auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                            auto tmp27 = tmp26 * (tmp26>0);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp29 = tmp4 ? tmp7 : tmp28;
                        out_ptr0[static_cast<long>(x3 + (49L*x1) + (98L*x2) + (22736L*x0))] = tmp29;
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(464L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (464L*x2) + (22736L*x0)), static_cast<long>(464L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (22736L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (464L*x2) + (22736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
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
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (24, ), (1, ))
    assert_size_stride(arg3_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (24, ), (1, ))
    assert_size_stride(arg6_1, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg7_1, (58, ), (1, ))
    assert_size_stride(arg8_1, (58, ), (1, ))
    assert_size_stride(arg9_1, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg10_1, (58, ), (1, ))
    assert_size_stride(arg11_1, (58, ), (1, ))
    assert_size_stride(arg12_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (58, ), (1, ))
    assert_size_stride(arg14_1, (58, ), (1, ))
    assert_size_stride(arg15_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg16_1, (58, ), (1, ))
    assert_size_stride(arg17_1, (58, ), (1, ))
    assert_size_stride(arg18_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg19_1, (58, ), (1, ))
    assert_size_stride(arg20_1, (58, ), (1, ))
    assert_size_stride(arg21_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (58, ), (1, ))
    assert_size_stride(arg23_1, (58, ), (1, ))
    assert_size_stride(arg24_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg25_1, (58, ), (1, ))
    assert_size_stride(arg26_1, (58, ), (1, ))
    assert_size_stride(arg27_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg28_1, (58, ), (1, ))
    assert_size_stride(arg29_1, (58, ), (1, ))
    assert_size_stride(arg30_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg31_1, (58, ), (1, ))
    assert_size_stride(arg32_1, (58, ), (1, ))
    assert_size_stride(arg33_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg34_1, (58, ), (1, ))
    assert_size_stride(arg35_1, (58, ), (1, ))
    assert_size_stride(arg36_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg37_1, (58, ), (1, ))
    assert_size_stride(arg38_1, (58, ), (1, ))
    assert_size_stride(arg39_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg40_1, (58, ), (1, ))
    assert_size_stride(arg41_1, (58, ), (1, ))
    assert_size_stride(arg42_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg43_1, (58, ), (1, ))
    assert_size_stride(arg44_1, (58, ), (1, ))
    assert_size_stride(arg45_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg46_1, (116, ), (1, ))
    assert_size_stride(arg47_1, (116, ), (1, ))
    assert_size_stride(arg48_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg49_1, (116, ), (1, ))
    assert_size_stride(arg50_1, (116, ), (1, ))
    assert_size_stride(arg51_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg52_1, (116, ), (1, ))
    assert_size_stride(arg53_1, (116, ), (1, ))
    assert_size_stride(arg54_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg55_1, (116, ), (1, ))
    assert_size_stride(arg56_1, (116, ), (1, ))
    assert_size_stride(arg57_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg58_1, (116, ), (1, ))
    assert_size_stride(arg59_1, (116, ), (1, ))
    assert_size_stride(arg60_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg61_1, (116, ), (1, ))
    assert_size_stride(arg62_1, (116, ), (1, ))
    assert_size_stride(arg63_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg64_1, (116, ), (1, ))
    assert_size_stride(arg65_1, (116, ), (1, ))
    assert_size_stride(arg66_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg67_1, (116, ), (1, ))
    assert_size_stride(arg68_1, (116, ), (1, ))
    assert_size_stride(arg69_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg70_1, (116, ), (1, ))
    assert_size_stride(arg71_1, (116, ), (1, ))
    assert_size_stride(arg72_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (116, ), (1, ))
    assert_size_stride(arg74_1, (116, ), (1, ))
    assert_size_stride(arg75_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg76_1, (116, ), (1, ))
    assert_size_stride(arg77_1, (116, ), (1, ))
    assert_size_stride(arg78_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg79_1, (116, ), (1, ))
    assert_size_stride(arg80_1, (116, ), (1, ))
    assert_size_stride(arg81_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg82_1, (116, ), (1, ))
    assert_size_stride(arg83_1, (116, ), (1, ))
    assert_size_stride(arg84_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg85_1, (116, ), (1, ))
    assert_size_stride(arg86_1, (116, ), (1, ))
    assert_size_stride(arg87_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg88_1, (116, ), (1, ))
    assert_size_stride(arg89_1, (116, ), (1, ))
    assert_size_stride(arg90_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg91_1, (116, ), (1, ))
    assert_size_stride(arg92_1, (116, ), (1, ))
    assert_size_stride(arg93_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg94_1, (116, ), (1, ))
    assert_size_stride(arg95_1, (116, ), (1, ))
    assert_size_stride(arg96_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg97_1, (116, ), (1, ))
    assert_size_stride(arg98_1, (116, ), (1, ))
    assert_size_stride(arg99_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (116, ), (1, ))
    assert_size_stride(arg101_1, (116, ), (1, ))
    assert_size_stride(arg102_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg103_1, (116, ), (1, ))
    assert_size_stride(arg104_1, (116, ), (1, ))
    assert_size_stride(arg105_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg106_1, (116, ), (1, ))
    assert_size_stride(arg107_1, (116, ), (1, ))
    assert_size_stride(arg108_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (116, ), (1, ))
    assert_size_stride(arg110_1, (116, ), (1, ))
    assert_size_stride(arg111_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg112_1, (116, ), (1, ))
    assert_size_stride(arg113_1, (116, ), (1, ))
    assert_size_stride(arg114_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg115_1, (116, ), (1, ))
    assert_size_stride(arg116_1, (116, ), (1, ))
    assert_size_stride(arg117_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (116, ), (1, ))
    assert_size_stride(arg119_1, (116, ), (1, ))
    assert_size_stride(arg120_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg121_1, (116, ), (1, ))
    assert_size_stride(arg122_1, (116, ), (1, ))
    assert_size_stride(arg123_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (232, ), (1, ))
    assert_size_stride(arg125_1, (232, ), (1, ))
    assert_size_stride(arg126_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg127_1, (232, ), (1, ))
    assert_size_stride(arg128_1, (232, ), (1, ))
    assert_size_stride(arg129_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg130_1, (232, ), (1, ))
    assert_size_stride(arg131_1, (232, ), (1, ))
    assert_size_stride(arg132_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (232, ), (1, ))
    assert_size_stride(arg134_1, (232, ), (1, ))
    assert_size_stride(arg135_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg136_1, (232, ), (1, ))
    assert_size_stride(arg137_1, (232, ), (1, ))
    assert_size_stride(arg138_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg139_1, (232, ), (1, ))
    assert_size_stride(arg140_1, (232, ), (1, ))
    assert_size_stride(arg141_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (232, ), (1, ))
    assert_size_stride(arg143_1, (232, ), (1, ))
    assert_size_stride(arg144_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg145_1, (232, ), (1, ))
    assert_size_stride(arg146_1, (232, ), (1, ))
    assert_size_stride(arg147_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg148_1, (232, ), (1, ))
    assert_size_stride(arg149_1, (232, ), (1, ))
    assert_size_stride(arg150_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (232, ), (1, ))
    assert_size_stride(arg152_1, (232, ), (1, ))
    assert_size_stride(arg153_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg154_1, (232, ), (1, ))
    assert_size_stride(arg155_1, (232, ), (1, ))
    assert_size_stride(arg156_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg157_1, (232, ), (1, ))
    assert_size_stride(arg158_1, (232, ), (1, ))
    assert_size_stride(arg159_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (232, ), (1, ))
    assert_size_stride(arg161_1, (232, ), (1, ))
    assert_size_stride(arg162_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg163_1, (232, ), (1, ))
    assert_size_stride(arg164_1, (232, ), (1, ))
    assert_size_stride(arg165_1, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1000, ), (1, ))
    assert_size_stride(arg170_1, (24, ), (1, ))
    assert_size_stride(arg171_1, (24, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (24, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (58, ), (1, ))
    assert_size_stride(arg177_1, (58, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (58, ), (1, ))
    assert_size_stride(arg180_1, (58, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (58, ), (1, ))
    assert_size_stride(arg183_1, (58, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (58, ), (1, ))
    assert_size_stride(arg186_1, (58, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (58, ), (1, ))
    assert_size_stride(arg189_1, (58, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (58, ), (1, ))
    assert_size_stride(arg192_1, (58, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (58, ), (1, ))
    assert_size_stride(arg195_1, (58, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (58, ), (1, ))
    assert_size_stride(arg198_1, (58, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (58, ), (1, ))
    assert_size_stride(arg201_1, (58, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (58, ), (1, ))
    assert_size_stride(arg204_1, (58, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (58, ), (1, ))
    assert_size_stride(arg207_1, (58, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (58, ), (1, ))
    assert_size_stride(arg210_1, (58, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (58, ), (1, ))
    assert_size_stride(arg213_1, (58, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (116, ), (1, ))
    assert_size_stride(arg216_1, (116, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (116, ), (1, ))
    assert_size_stride(arg219_1, (116, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (116, ), (1, ))
    assert_size_stride(arg222_1, (116, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (116, ), (1, ))
    assert_size_stride(arg225_1, (116, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (116, ), (1, ))
    assert_size_stride(arg228_1, (116, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (116, ), (1, ))
    assert_size_stride(arg231_1, (116, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (116, ), (1, ))
    assert_size_stride(arg234_1, (116, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (116, ), (1, ))
    assert_size_stride(arg237_1, (116, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (116, ), (1, ))
    assert_size_stride(arg240_1, (116, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (116, ), (1, ))
    assert_size_stride(arg243_1, (116, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (116, ), (1, ))
    assert_size_stride(arg246_1, (116, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (116, ), (1, ))
    assert_size_stride(arg249_1, (116, ), (1, ))
    assert_size_stride(arg250_1, (), ())
    assert_size_stride(arg251_1, (116, ), (1, ))
    assert_size_stride(arg252_1, (116, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (116, ), (1, ))
    assert_size_stride(arg255_1, (116, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (116, ), (1, ))
    assert_size_stride(arg258_1, (116, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (116, ), (1, ))
    assert_size_stride(arg261_1, (116, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (116, ), (1, ))
    assert_size_stride(arg264_1, (116, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (116, ), (1, ))
    assert_size_stride(arg267_1, (116, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (116, ), (1, ))
    assert_size_stride(arg270_1, (116, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (116, ), (1, ))
    assert_size_stride(arg273_1, (116, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (116, ), (1, ))
    assert_size_stride(arg276_1, (116, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (116, ), (1, ))
    assert_size_stride(arg279_1, (116, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (116, ), (1, ))
    assert_size_stride(arg282_1, (116, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (116, ), (1, ))
    assert_size_stride(arg285_1, (116, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (116, ), (1, ))
    assert_size_stride(arg288_1, (116, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (116, ), (1, ))
    assert_size_stride(arg291_1, (116, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (232, ), (1, ))
    assert_size_stride(arg294_1, (232, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (232, ), (1, ))
    assert_size_stride(arg297_1, (232, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (232, ), (1, ))
    assert_size_stride(arg300_1, (232, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (232, ), (1, ))
    assert_size_stride(arg303_1, (232, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (232, ), (1, ))
    assert_size_stride(arg306_1, (232, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (232, ), (1, ))
    assert_size_stride(arg309_1, (232, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (232, ), (1, ))
    assert_size_stride(arg312_1, (232, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (232, ), (1, ))
    assert_size_stride(arg315_1, (232, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (232, ), (1, ))
    assert_size_stride(arg318_1, (232, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (232, ), (1, ))
    assert_size_stride(arg321_1, (232, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (232, ), (1, ))
    assert_size_stride(arg324_1, (232, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (232, ), (1, ))
    assert_size_stride(arg327_1, (232, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (232, ), (1, ))
    assert_size_stride(arg330_1, (232, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (232, ), (1, ))
    assert_size_stride(arg333_1, (232, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg338_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg338_1
    # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 24, 112, 112), (301056, 1, 2688, 24))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg170_1
    del arg171_1
    del arg1_1
    del arg2_1
    del buf3
    # Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf5, (4, 24, 28, 28), (18816, 1, 672, 24))
    del arg3_1
    buf6 = buf5; del buf5  # reuse
    cpp_fused__native_batch_norm_legit_no_training_2(c_void_p(buf6.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg173_1
    del arg174_1
    del arg4_1
    del arg5_1
    # Source Nodes: [getattr_l__mod___stage2___0___branch1_1, getattr_l__mod___stage2___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf7 = extern_kernels.convolution(buf6, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf7, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg6_1
    del buf6
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf4, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 58, 56, 56), (181888, 1, 3248, 58))
    del arg9_1
    del buf4
    buf9 = buf8; del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg179_1
    del arg180_1
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2, getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf10 = extern_kernels.convolution(buf9, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf10, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg12_1
    del buf9
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_4(c_void_p(buf11.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg13_1
    del arg14_1
    del arg182_1
    del arg183_1
    # Source Nodes: [getattr_l__mod___stage2___0___branch2_4, getattr_l__mod___stage2___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf12 = extern_kernels.convolution(buf11, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg15_1
    buf13 = empty((4, 116, 28, 28), device='cpu', dtype=torch.float32)
    buf14 = empty((4, 58, 2, 28, 28), device='cpu', dtype=torch.float32)
    buf15 = buf11; del buf11  # reuse
    cpp_fused_cat_clone_convolution_5(c_void_p(buf7.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg16_1
    del arg176_1
    del arg177_1
    del arg17_1
    del arg185_1
    del arg186_1
    del arg7_1
    del arg8_1
    del buf12
    del buf7
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg18_1
    del buf15
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf17.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()))
    del arg188_1
    del arg189_1
    del arg19_1
    del arg20_1
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2, getattr_l__mod___stage2___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf18 = extern_kernels.convolution(buf17, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf18, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg21_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_7(c_void_p(buf19.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg191_1
    del arg192_1
    del arg22_1
    del arg23_1
    # Source Nodes: [getattr_l__mod___stage2___1___branch2_4, getattr_l__mod___stage2___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf20 = extern_kernels.convolution(buf19, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg24_1
    buf21 = reinterpret_tensor(buf13, (4, 58, 2, 28, 28), (90944, 1568, 784, 28, 1), 0); del buf13  # reuse
    buf22 = buf19; del buf19  # reuse
    cpp_fused_clone_convolution_8(c_void_p(buf14.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg194_1
    del arg195_1
    del arg25_1
    del arg26_1
    del buf20
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(buf22, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg27_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf24.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg197_1
    del arg198_1
    del arg28_1
    del arg29_1
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_1, getattr_l__mod___stage2___2___branch2_2, getattr_l__mod___stage2___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf25 = extern_kernels.convolution(buf24, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf25, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg30_1
    del buf24
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_10(c_void_p(buf26.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg200_1
    del arg201_1
    del arg31_1
    del arg32_1
    # Source Nodes: [getattr_l__mod___stage2___2___branch2_4, getattr_l__mod___stage2___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf27 = extern_kernels.convolution(buf26, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg33_1
    buf28 = buf14; del buf14  # reuse
    buf29 = buf26; del buf26  # reuse
    cpp_fused_clone_convolution_11(c_void_p(buf21.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg203_1
    del arg204_1
    del arg34_1
    del arg35_1
    del buf27
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg36_1
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf31.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg37_1
    del arg38_1
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_1, getattr_l__mod___stage2___3___branch2_2, getattr_l__mod___stage2___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf32 = extern_kernels.convolution(buf31, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
    assert_size_stride(buf32, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg39_1
    buf33 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_13(c_void_p(buf33.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg209_1
    del arg210_1
    del arg40_1
    del arg41_1
    # Source Nodes: [getattr_l__mod___stage2___3___branch2_4, getattr_l__mod___stage2___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf34 = extern_kernels.convolution(buf33, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (4, 58, 28, 28), (45472, 1, 1624, 58))
    del arg42_1
    buf35 = buf21; del buf21  # reuse
    buf36 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_14(c_void_p(buf28.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf40.data_ptr()))
    del arg212_1
    del arg213_1
    del arg43_1
    del arg44_1
    del buf28
    del buf35
    # Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf37, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg45_1
    del buf36
    buf38 = buf37; del buf37  # reuse
    cpp_fused__native_batch_norm_legit_no_training_15(c_void_p(buf38.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg215_1
    del arg216_1
    del arg46_1
    del arg47_1
    # Source Nodes: [getattr_l__mod___stage3___0___branch1_1, getattr_l__mod___stage3___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf39 = extern_kernels.convolution(buf38, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg48_1
    del buf38
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf40, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf41, (4, 116, 28, 28), (90944, 1, 3248, 116))
    del arg51_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_16(c_void_p(buf42.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg52_1
    del arg53_1
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2, getattr_l__mod___stage3___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf43 = extern_kernels.convolution(buf42, arg54_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf43, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg54_1
    del buf42
    buf44 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_17(c_void_p(buf44.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()))
    del arg224_1
    del arg225_1
    del arg55_1
    del arg56_1
    # Source Nodes: [getattr_l__mod___stage3___0___branch2_4, getattr_l__mod___stage3___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf45 = extern_kernels.convolution(buf44, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg57_1
    buf46 = reinterpret_tensor(buf34, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf34  # reuse
    buf47 = reinterpret_tensor(buf33, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf33  # reuse
    buf48 = buf44; del buf44  # reuse
    cpp_fused_cat_clone_convolution_18(c_void_p(buf39.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg218_1
    del arg219_1
    del arg227_1
    del arg228_1
    del arg49_1
    del arg50_1
    del arg58_1
    del arg59_1
    del buf39
    del buf45
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg60_1
    del buf48
    buf50 = buf49; del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_19(c_void_p(buf50.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()))
    del arg230_1
    del arg231_1
    del arg61_1
    del arg62_1
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2, getattr_l__mod___stage3___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf51 = extern_kernels.convolution(buf50, arg63_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf51, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg63_1
    del buf50
    buf52 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_20(c_void_p(buf52.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg233_1
    del arg234_1
    del arg64_1
    del arg65_1
    # Source Nodes: [getattr_l__mod___stage3___1___branch2_4, getattr_l__mod___stage3___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf53 = extern_kernels.convolution(buf52, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf53, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg66_1
    buf54 = reinterpret_tensor(buf46, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf46  # reuse
    buf55 = buf52; del buf52  # reuse
    cpp_fused_clone_convolution_21(c_void_p(buf47.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg236_1
    del arg237_1
    del arg67_1
    del arg68_1
    del buf53
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg69_1
    del buf55
    buf57 = buf56; del buf56  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_22(c_void_p(buf57.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg70_1
    del arg71_1
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_1, getattr_l__mod___stage3___2___branch2_2, getattr_l__mod___stage3___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf57, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf58, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg72_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_23(c_void_p(buf59.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg73_1
    del arg74_1
    # Source Nodes: [getattr_l__mod___stage3___2___branch2_4, getattr_l__mod___stage3___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg75_1
    buf61 = buf47; del buf47  # reuse
    buf62 = buf59; del buf59  # reuse
    cpp_fused_clone_convolution_24(c_void_p(buf54.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg245_1
    del arg246_1
    del arg76_1
    del arg77_1
    del buf60
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg78_1
    del buf62
    buf64 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_25(c_void_p(buf64.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg248_1
    del arg249_1
    del arg79_1
    del arg80_1
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_1, getattr_l__mod___stage3___3___branch2_2, getattr_l__mod___stage3___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf65 = extern_kernels.convolution(buf64, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf65, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg81_1
    del buf64
    buf66 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_26(c_void_p(buf66.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg82_1
    del arg83_1
    # Source Nodes: [getattr_l__mod___stage3___3___branch2_4, getattr_l__mod___stage3___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf67 = extern_kernels.convolution(buf66, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg84_1
    buf68 = buf54; del buf54  # reuse
    buf69 = buf66; del buf66  # reuse
    cpp_fused_clone_convolution_27(c_void_p(buf61.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg254_1
    del arg255_1
    del arg85_1
    del arg86_1
    del buf67
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg87_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_28(c_void_p(buf71.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg257_1
    del arg258_1
    del arg88_1
    del arg89_1
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_1, getattr_l__mod___stage3___4___branch2_2, getattr_l__mod___stage3___4___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf72 = extern_kernels.convolution(buf71, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf72, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg90_1
    del buf71
    buf73 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_29(c_void_p(buf73.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()))
    del arg260_1
    del arg261_1
    del arg91_1
    del arg92_1
    # Source Nodes: [getattr_l__mod___stage3___4___branch2_4, getattr_l__mod___stage3___4___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf74 = extern_kernels.convolution(buf73, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg93_1
    buf75 = buf61; del buf61  # reuse
    buf76 = buf73; del buf73  # reuse
    cpp_fused_clone_convolution_30(c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg263_1
    del arg264_1
    del arg94_1
    del arg95_1
    del buf74
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
    buf77 = extern_kernels.convolution(buf76, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg96_1
    del buf76
    buf78 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_31(c_void_p(buf78.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()))
    del arg266_1
    del arg267_1
    del arg97_1
    del arg98_1
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_1, getattr_l__mod___stage3___5___branch2_2, getattr_l__mod___stage3___5___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf79 = extern_kernels.convolution(buf78, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf79, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg99_1
    del buf78
    buf80 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_32(c_void_p(buf80.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg269_1
    del arg270_1
    # Source Nodes: [getattr_l__mod___stage3___5___branch2_4, getattr_l__mod___stage3___5___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf81 = extern_kernels.convolution(buf80, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg102_1
    buf82 = buf68; del buf68  # reuse
    buf83 = buf80; del buf80  # reuse
    cpp_fused_clone_convolution_33(c_void_p(buf75.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg103_1
    del arg104_1
    del arg272_1
    del arg273_1
    del buf81
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg105_1
    del buf83
    buf85 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_34(c_void_p(buf85.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg275_1
    del arg276_1
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_1, getattr_l__mod___stage3___6___branch2_2, getattr_l__mod___stage3___6___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf86 = extern_kernels.convolution(buf85, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf86, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg108_1
    del buf85
    buf87 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_35(c_void_p(buf87.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg278_1
    del arg279_1
    # Source Nodes: [getattr_l__mod___stage3___6___branch2_4, getattr_l__mod___stage3___6___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg111_1
    buf89 = buf75; del buf75  # reuse
    buf90 = buf87; del buf87  # reuse
    cpp_fused_clone_convolution_36(c_void_p(buf82.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg112_1
    del arg113_1
    del arg281_1
    del arg282_1
    del buf88
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg114_1
    del buf90
    buf92 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf92.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg284_1
    del arg285_1
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_1, getattr_l__mod___stage3___7___branch2_2, getattr_l__mod___stage3___7___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf93 = extern_kernels.convolution(buf92, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
    assert_size_stride(buf93, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg117_1
    buf94 = buf93; del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_38(c_void_p(buf94.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg287_1
    del arg288_1
    # Source Nodes: [getattr_l__mod___stage3___7___branch2_4, getattr_l__mod___stage3___7___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (4, 116, 14, 14), (22736, 1, 1624, 116))
    del arg120_1
    buf96 = buf82; del buf82  # reuse
    buf97 = reinterpret_tensor(buf31, (4, 232, 14, 14), (45472, 1, 3248, 232), 0); del buf31  # reuse
    buf101 = reinterpret_tensor(buf29, (4, 232, 14, 14), (45472, 1, 3248, 232), 0); del buf29  # reuse
    cpp_fused_clone_convolution_39(c_void_p(buf89.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg121_1
    del arg122_1
    del arg290_1
    del arg291_1
    del buf89
    del buf96
    # Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, arg123_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf98, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg123_1
    del buf97
    buf99 = buf98; del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_40(c_void_p(buf99.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg293_1
    del arg294_1
    # Source Nodes: [getattr_l__mod___stage4___0___branch1_1, getattr_l__mod___stage4___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf100 = extern_kernels.convolution(buf99, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg126_1
    del buf99
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 232, 14, 14), (45472, 1, 3248, 232))
    del arg129_1
    del buf101
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_41(c_void_p(buf103.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()))
    del arg130_1
    del arg131_1
    del arg299_1
    del arg300_1
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2, getattr_l__mod___stage4___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf104 = extern_kernels.convolution(buf103, arg132_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf104, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg132_1
    del buf103
    buf105 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_42(c_void_p(buf105.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg302_1
    del arg303_1
    # Source Nodes: [getattr_l__mod___stage4___0___branch2_4, getattr_l__mod___stage4___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg135_1
    buf107 = reinterpret_tensor(buf95, (4, 464, 7, 7), (22736, 49, 7, 1), 0); del buf95  # reuse
    buf108 = reinterpret_tensor(buf94, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf94  # reuse
    buf109 = buf105; del buf105  # reuse
    cpp_fused_cat_clone_convolution_43(c_void_p(buf100.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg127_1
    del arg128_1
    del arg136_1
    del arg137_1
    del arg296_1
    del arg297_1
    del arg305_1
    del arg306_1
    del buf100
    del buf106
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(buf109, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg138_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf111.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg308_1
    del arg309_1
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2, getattr_l__mod___stage4___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf111, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf112, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg141_1
    del buf111
    buf113 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_45(c_void_p(buf113.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg311_1
    del arg312_1
    # Source Nodes: [getattr_l__mod___stage4___1___branch2_4, getattr_l__mod___stage4___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf114 = extern_kernels.convolution(buf113, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg144_1
    buf115 = reinterpret_tensor(buf107, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf107  # reuse
    buf116 = buf113; del buf113  # reuse
    cpp_fused_clone_convolution_46(c_void_p(buf108.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg145_1
    del arg146_1
    del arg314_1
    del arg315_1
    del buf114
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
    buf117 = extern_kernels.convolution(buf116, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf117, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg147_1
    del buf116
    buf118 = buf117; del buf117  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf118.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()))
    del arg148_1
    del arg149_1
    del arg317_1
    del arg318_1
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_1, getattr_l__mod___stage4___2___branch2_2, getattr_l__mod___stage4___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf119 = extern_kernels.convolution(buf118, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf119, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg150_1
    del buf118
    buf120 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_48(c_void_p(buf120.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg320_1
    del arg321_1
    # Source Nodes: [getattr_l__mod___stage4___2___branch2_4, getattr_l__mod___stage4___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf121 = extern_kernels.convolution(buf120, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf121, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg153_1
    buf122 = buf108; del buf108  # reuse
    buf123 = buf120; del buf120  # reuse
    cpp_fused_clone_convolution_49(c_void_p(buf115.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg154_1
    del arg155_1
    del arg323_1
    del arg324_1
    del buf121
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg156_1
    del buf123
    buf125 = buf124; del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf125.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()))
    del arg157_1
    del arg158_1
    del arg326_1
    del arg327_1
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_1, getattr_l__mod___stage4___3___branch2_2, getattr_l__mod___stage4___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf125, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
    assert_size_stride(buf126, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg159_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_51(c_void_p(buf127.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()))
    del arg160_1
    del arg161_1
    del arg329_1
    del arg330_1
    # Source Nodes: [getattr_l__mod___stage4___3___branch2_4, getattr_l__mod___stage4___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf128 = extern_kernels.convolution(buf127, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (4, 232, 7, 7), (11368, 1, 1624, 232))
    del arg162_1
    del buf127
    buf129 = buf115; del buf115  # reuse
    buf130 = reinterpret_tensor(buf92, (4, 464, 7, 7), (22736, 1, 3248, 464), 0); del buf92  # reuse
    cpp_fused_clone_convolution_52(c_void_p(buf122.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg163_1
    del arg164_1
    del arg332_1
    del arg333_1
    del buf122
    del buf128
    del buf129
    # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    del arg165_1
    del buf130
    buf132 = empty((4, 1024), device='cpu', dtype=torch.float32)
    buf133 = buf132; del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_53(c_void_p(buf133.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()))
    del arg166_1
    del arg167_1
    del arg335_1
    del arg336_1
    del buf131
    buf134 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___conv5_1, x_53, x_54, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.addmm, aten.mean, aten.relu]
    extern_kernels.addmm(arg169_1, buf133, reinterpret_tensor(arg168_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf134)
    del arg168_1
    del arg169_1
    return (buf134, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg173_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg176_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg179_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg182_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg185_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg188_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg191_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg194_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg197_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg200_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg203_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg206_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg212_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg218_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg221_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg224_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg227_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg230_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg233_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg236_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg239_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg242_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg245_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg248_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg251_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg254_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg257_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg260_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg263_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg266_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg269_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg272_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg275_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg278_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg281_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg284_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg290_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg293_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg296_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg299_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg302_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg305_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg308_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg311_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg314_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg317_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg320_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg323_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg326_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg329_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg332_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg335_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg338_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
