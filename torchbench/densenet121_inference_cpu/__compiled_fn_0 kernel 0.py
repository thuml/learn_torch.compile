
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
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
                                auto tmp12 = masked_load(in_out_ptr0 + static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp10));
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
                                auto tmp20 = masked_load(in_out_ptr0 + static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp18));
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
                                auto tmp29 = masked_load(in_out_ptr0 + static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp27));
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
                                auto tmp38 = masked_load(in_out_ptr0 + static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = at::vec::maximum(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = at::vec::maximum(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_out_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp46));
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
                                auto tmp57 = masked_load(in_out_ptr0 + static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = at::vec::maximum(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_out_ptr0 + static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = at::vec::maximum(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_out_ptr0 + static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = at::vec::maximum(tmp68, tmp64);
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (192L*x2) + (10752L*x1) + (602112L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (602112L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                        tmp17.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200704L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(96);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = tmp28 * (tmp28>0);
                    out_ptr0[static_cast<long>(x1 + (96L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(96);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-96L) + x1 + (32L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp37;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(96);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-96L) + x1 + (32L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(160);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-128L) + x1 + (32L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_9 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr7 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr8 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr9 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr10 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr11 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (14336L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7296L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp12 + tmp14;
                        auto tmp16 = tmp15.sqrt();
                        auto tmp17 = tmp16.reciprocal();
                        auto tmp18 = static_cast<float>(1.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp11 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = at::vec::clamp_min(tmp25, decltype(tmp25)(0));
                        tmp26.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (3584L*x0)));
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr2[static_cast<long>(x2)];
                        auto tmp25 = in_ptr3[static_cast<long>(x2)];
                        auto tmp33 = in_ptr4[static_cast<long>(x2)];
                        auto tmp35 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(128);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(128L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7296L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(160);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr1[static_cast<long>((-128L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp4 ? tmp15 : tmp21;
                        auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = std::sqrt(tmp27);
                        auto tmp29 = 1 / tmp28;
                        auto tmp30 = static_cast<float>(1.0);
                        auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                        auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                        auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                        auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                        out_ptr0[static_cast<long>(x2 + (160L*x1) + (4480L*x0))] = tmp36;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x2)];
                        auto tmp33 = in_ptr4[static_cast<long>(x2)];
                        auto tmp41 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(128);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(128L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7296L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(160);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((-128L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp0 >= tmp17;
                        auto tmp24 = static_cast<long>(192);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr2[static_cast<long>((-160L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp4 ? tmp15 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(1e-05);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (5376L*x0))] = tmp42;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(1L))
                    {
                        auto tmp39 = in_ptr4[static_cast<long>(x2)];
                        auto tmp41 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(128);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(128L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7296L + x2 + (256L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(160);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((-128L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp0 >= tmp17;
                        auto tmp24 = static_cast<long>(192);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = tmp23 & tmp25;
                        auto tmp27 = [&]
                        {
                            auto tmp28 = in_ptr2[static_cast<long>((-160L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                        auto tmp30 = tmp0 >= tmp24;
                        auto tmp31 = static_cast<long>(224);
                        auto tmp32 = tmp0 < tmp31;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr3[static_cast<long>((-192L) + x2 + (32L*x1) + (896L*x0))];
                            return tmp34;
                        }
                        ;
                        auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                        auto tmp36 = tmp26 ? tmp29 : tmp35;
                        auto tmp37 = tmp19 ? tmp22 : tmp36;
                        auto tmp38 = tmp4 ? tmp15 : tmp37;
                        auto tmp40 = decltype(tmp38)(tmp38 - tmp39);
                        auto tmp42 = static_cast<float>(1e-05);
                        auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                        auto tmp44 = std::sqrt(tmp43);
                        auto tmp45 = 1 / tmp44;
                        auto tmp46 = static_cast<float>(1.0);
                        auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                        auto tmp48 = decltype(tmp40)(tmp40 * tmp47);
                        out_ptr0[static_cast<long>(x2 + (224L*x1) + (6272L*x0))] = tmp48;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (288L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (256L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (288L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (256L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (288L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr20 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (352L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (288L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (320L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (352L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (416L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_26 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (448L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (352L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_30 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_32 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(416L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_38 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (14336L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7424L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp12 + tmp14;
                        auto tmp16 = tmp15.sqrt();
                        auto tmp17 = tmp16.reciprocal();
                        auto tmp18 = static_cast<float>(1.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp11 * tmp20;
                        auto tmp23 = tmp21 * tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = at::vec::clamp_min(tmp25, decltype(tmp25)(0));
                        tmp26.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (3584L*x0)));
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (384L*x1) + (5376L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr2[static_cast<long>(x2)];
                        auto tmp25 = in_ptr3[static_cast<long>(x2)];
                        auto tmp33 = in_ptr4[static_cast<long>(x2)];
                        auto tmp35 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(256);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(256L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7424L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(288);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr1[static_cast<long>((-256L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp4 ? tmp15 : tmp21;
                        auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = std::sqrt(tmp27);
                        auto tmp29 = 1 / tmp28;
                        auto tmp30 = static_cast<float>(1.0);
                        auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                        auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                        auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                        auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                        out_ptr0[static_cast<long>(x2 + (288L*x1) + (4032L*x0))] = tmp36;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(225792L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr3[static_cast<long>(x2)];
                        auto tmp33 = in_ptr4[static_cast<long>(x2)];
                        auto tmp41 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(256);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(256L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7424L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(288);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((-256L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp0 >= tmp17;
                        auto tmp24 = static_cast<long>(320);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr2[static_cast<long>((-288L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp4 ? tmp15 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(1e-05);
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        auto tmp36 = std::sqrt(tmp35);
                        auto tmp37 = 1 / tmp36;
                        auto tmp38 = static_cast<float>(1.0);
                        auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                        auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                        auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                        out_ptr0[static_cast<long>(x2 + (320L*x1) + (4480L*x0))] = tmp42;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(352L); x2+=static_cast<long>(1L))
                    {
                        auto tmp39 = in_ptr4[static_cast<long>(x2)];
                        auto tmp41 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(256);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(256L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7424L + x2 + (512L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(288);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((-256L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp0 >= tmp17;
                        auto tmp24 = static_cast<long>(320);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = tmp23 & tmp25;
                        auto tmp27 = [&]
                        {
                            auto tmp28 = in_ptr2[static_cast<long>((-288L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                        auto tmp30 = tmp0 >= tmp24;
                        auto tmp31 = static_cast<long>(352);
                        auto tmp32 = tmp0 < tmp31;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr3[static_cast<long>((-320L) + x2 + (32L*x1) + (448L*x0))];
                            return tmp34;
                        }
                        ;
                        auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                        auto tmp36 = tmp26 ? tmp29 : tmp35;
                        auto tmp37 = tmp19 ? tmp22 : tmp36;
                        auto tmp38 = tmp4 ? tmp15 : tmp37;
                        auto tmp40 = decltype(tmp38)(tmp38 - tmp39);
                        auto tmp42 = static_cast<float>(1e-05);
                        auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                        auto tmp44 = std::sqrt(tmp43);
                        auto tmp45 = 1 / tmp44;
                        auto tmp46 = static_cast<float>(1.0);
                        auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                        auto tmp48 = decltype(tmp40)(tmp40 * tmp47);
                        out_ptr0[static_cast<long>(x2 + (352L*x1) + (4928L*x0))] = tmp48;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_47 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr20 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_49 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (544L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (416L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (512L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (544L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(416L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_51 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (576L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_53 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (608L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (608L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_57 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(544L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_59 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (608L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_61 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (704L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (704L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(608L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_63 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (736L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (736L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_65 = async_compile.cpp('''
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
                       const float* in_ptr9,
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr20 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr21 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr22 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr23 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_67 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (736L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_69 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_71 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_73 = async_compile.cpp('''
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
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (864L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (800L*x0)));
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr32 + static_cast<long>(x1 + (864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_75 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_77 = async_compile.cpp('''
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
                       float* out_ptr14)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(864L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_79 = async_compile.cpp('''
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
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32,
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (928L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr32 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr33 + static_cast<long>(x1 + (896L*x0)));
                        tmp0.store(out_ptr34 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr35 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_81 = async_compile.cpp('''
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
                       float* out_ptr14)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(928L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_83 = async_compile.cpp('''
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
                       const float* in_ptr9,
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
                       float* out_ptr17)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_85 = async_compile.cpp('''
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
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (992L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(992L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_87 = async_compile.cpp('''
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
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.sqrt();
                    auto tmp17 = tmp16.reciprocal();
                    auto tmp18 = static_cast<float>(1.0);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp11 * tmp20;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = at::vec::clamp_min(tmp25, decltype(tmp25)(0));
                    tmp26.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (3584L*x0)));
                    tmp9.store(out_ptr1 + static_cast<long>(x2 + (640L*x1) + (4480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(544L); x2+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr2[static_cast<long>(x2)];
                    auto tmp25 = in_ptr3[static_cast<long>(x2)];
                    auto tmp33 = in_ptr4[static_cast<long>(x2)];
                    auto tmp35 = in_ptr5[static_cast<long>(x2)];
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp7 = in_ptr0[static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                        auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                        auto tmp11 = in_ptr0[static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                        auto tmp13 = static_cast<float>(0.25);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp16 = tmp0 >= tmp3;
                    auto tmp17 = static_cast<long>(544);
                    auto tmp18 = tmp0 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr1[static_cast<long>((-512L) + x2 + (32L*x1) + (224L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp16 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp4 ? tmp15 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    out_ptr0[static_cast<long>(x2 + (544L*x1) + (3808L*x0))] = tmp36;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(106624L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr3[static_cast<long>(x2)];
                    auto tmp33 = in_ptr4[static_cast<long>(x2)];
                    auto tmp41 = in_ptr5[static_cast<long>(x2)];
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp7 = in_ptr0[static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                        auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                        auto tmp11 = in_ptr0[static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0))];
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                        auto tmp13 = static_cast<float>(0.25);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp16 = tmp0 >= tmp3;
                    auto tmp17 = static_cast<long>(544);
                    auto tmp18 = tmp0 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr1[static_cast<long>((-512L) + x2 + (32L*x1) + (224L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp0 >= tmp17;
                    auto tmp24 = static_cast<long>(576);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr2[static_cast<long>((-544L) + x2 + (32L*x1) + (224L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp4 ? tmp15 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    out_ptr0[static_cast<long>(x2 + (576L*x1) + (4032L*x0))] = tmp42;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(608L); x2+=static_cast<long>(1L))
                    {
                        auto tmp39 = in_ptr4[static_cast<long>(x2)];
                        auto tmp41 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (14336L*x0))];
                            auto tmp7 = in_ptr0[static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0))];
                            auto tmp8 = decltype(tmp7)(tmp7 + tmp6);
                            auto tmp9 = in_ptr0[static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0))];
                            auto tmp10 = decltype(tmp9)(tmp9 + tmp8);
                            auto tmp11 = in_ptr0[static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0))];
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp10);
                            auto tmp13 = static_cast<float>(0.25);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp16 = tmp0 >= tmp3;
                        auto tmp17 = static_cast<long>(544);
                        auto tmp18 = tmp0 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((-512L) + x2 + (32L*x1) + (224L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp0 >= tmp17;
                        auto tmp24 = static_cast<long>(576);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = tmp23 & tmp25;
                        auto tmp27 = [&]
                        {
                            auto tmp28 = in_ptr2[static_cast<long>((-544L) + x2 + (32L*x1) + (224L*x0))];
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                        auto tmp30 = tmp0 >= tmp24;
                        auto tmp31 = static_cast<long>(608);
                        auto tmp32 = tmp0 < tmp31;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr3[static_cast<long>((-576L) + x2 + (32L*x1) + (224L*x0))];
                            return tmp34;
                        }
                        ;
                        auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                        auto tmp36 = tmp26 ? tmp29 : tmp35;
                        auto tmp37 = tmp19 ? tmp22 : tmp36;
                        auto tmp38 = tmp4 ? tmp15 : tmp37;
                        auto tmp40 = decltype(tmp38)(tmp38 - tmp39);
                        auto tmp42 = static_cast<float>(1e-05);
                        auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                        auto tmp44 = std::sqrt(tmp43);
                        auto tmp45 = 1 / tmp44;
                        auto tmp46 = static_cast<float>(1.0);
                        auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                        auto tmp48 = decltype(tmp40)(tmp40 * tmp47);
                        out_ptr0[static_cast<long>(x2 + (608L*x1) + (4256L*x0))] = tmp48;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(608L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_96 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr20 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_98 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_100 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_102 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_104 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_106 = async_compile.cpp('''
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
                       float* out_ptr18,
                       float* out_ptr19)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_108 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_110 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(864L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_112 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_114 = async_compile.cpp('''
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
                       const float* in_ptr9,
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
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr20 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr21 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr22 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr23 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(928L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_116 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_118 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(992L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_120 = async_compile.cpp('''
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
    auto out_ptr4 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (160, ), (1, ))
    assert_size_stride(arg23_1, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg33_1, (224, ), (1, ))
    assert_size_stride(arg34_1, (224, ), (1, ))
    assert_size_stride(arg35_1, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg48_1, (160, ), (1, ))
    assert_size_stride(arg49_1, (160, ), (1, ))
    assert_size_stride(arg50_1, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg54_1, (192, ), (1, ))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg60_1, (224, ), (1, ))
    assert_size_stride(arg61_1, (224, ), (1, ))
    assert_size_stride(arg62_1, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg72_1, (288, ), (1, ))
    assert_size_stride(arg73_1, (288, ), (1, ))
    assert_size_stride(arg74_1, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg78_1, (320, ), (1, ))
    assert_size_stride(arg79_1, (320, ), (1, ))
    assert_size_stride(arg80_1, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg84_1, (352, ), (1, ))
    assert_size_stride(arg85_1, (352, ), (1, ))
    assert_size_stride(arg86_1, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg96_1, (416, ), (1, ))
    assert_size_stride(arg97_1, (416, ), (1, ))
    assert_size_stride(arg98_1, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg102_1, (448, ), (1, ))
    assert_size_stride(arg103_1, (448, ), (1, ))
    assert_size_stride(arg104_1, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg108_1, (480, ), (1, ))
    assert_size_stride(arg109_1, (480, ), (1, ))
    assert_size_stride(arg110_1, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg123_1, (288, ), (1, ))
    assert_size_stride(arg124_1, (288, ), (1, ))
    assert_size_stride(arg125_1, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg129_1, (320, ), (1, ))
    assert_size_stride(arg130_1, (320, ), (1, ))
    assert_size_stride(arg131_1, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg135_1, (352, ), (1, ))
    assert_size_stride(arg136_1, (352, ), (1, ))
    assert_size_stride(arg137_1, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg144_1, (128, ), (1, ))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg147_1, (416, ), (1, ))
    assert_size_stride(arg148_1, (416, ), (1, ))
    assert_size_stride(arg149_1, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg153_1, (448, ), (1, ))
    assert_size_stride(arg154_1, (448, ), (1, ))
    assert_size_stride(arg155_1, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg156_1, (128, ), (1, ))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (480, ), (1, ))
    assert_size_stride(arg161_1, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg171_1, (544, ), (1, ))
    assert_size_stride(arg172_1, (544, ), (1, ))
    assert_size_stride(arg173_1, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg177_1, (576, ), (1, ))
    assert_size_stride(arg178_1, (576, ), (1, ))
    assert_size_stride(arg179_1, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg183_1, (608, ), (1, ))
    assert_size_stride(arg184_1, (608, ), (1, ))
    assert_size_stride(arg185_1, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg186_1, (128, ), (1, ))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg189_1, (640, ), (1, ))
    assert_size_stride(arg190_1, (640, ), (1, ))
    assert_size_stride(arg191_1, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg192_1, (128, ), (1, ))
    assert_size_stride(arg193_1, (128, ), (1, ))
    assert_size_stride(arg194_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (672, ), (1, ))
    assert_size_stride(arg197_1, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg201_1, (704, ), (1, ))
    assert_size_stride(arg202_1, (704, ), (1, ))
    assert_size_stride(arg203_1, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg207_1, (736, ), (1, ))
    assert_size_stride(arg208_1, (736, ), (1, ))
    assert_size_stride(arg209_1, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg219_1, (800, ), (1, ))
    assert_size_stride(arg220_1, (800, ), (1, ))
    assert_size_stride(arg221_1, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg225_1, (832, ), (1, ))
    assert_size_stride(arg226_1, (832, ), (1, ))
    assert_size_stride(arg227_1, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg231_1, (864, ), (1, ))
    assert_size_stride(arg232_1, (864, ), (1, ))
    assert_size_stride(arg233_1, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg237_1, (896, ), (1, ))
    assert_size_stride(arg238_1, (896, ), (1, ))
    assert_size_stride(arg239_1, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg240_1, (128, ), (1, ))
    assert_size_stride(arg241_1, (128, ), (1, ))
    assert_size_stride(arg242_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg243_1, (928, ), (1, ))
    assert_size_stride(arg244_1, (928, ), (1, ))
    assert_size_stride(arg245_1, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg249_1, (960, ), (1, ))
    assert_size_stride(arg250_1, (960, ), (1, ))
    assert_size_stride(arg251_1, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg255_1, (992, ), (1, ))
    assert_size_stride(arg256_1, (992, ), (1, ))
    assert_size_stride(arg257_1, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg270_1, (544, ), (1, ))
    assert_size_stride(arg271_1, (544, ), (1, ))
    assert_size_stride(arg272_1, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(arg273_1, (128, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg276_1, (576, ), (1, ))
    assert_size_stride(arg277_1, (576, ), (1, ))
    assert_size_stride(arg278_1, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg282_1, (608, ), (1, ))
    assert_size_stride(arg283_1, (608, ), (1, ))
    assert_size_stride(arg284_1, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg288_1, (640, ), (1, ))
    assert_size_stride(arg289_1, (640, ), (1, ))
    assert_size_stride(arg290_1, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg294_1, (672, ), (1, ))
    assert_size_stride(arg295_1, (672, ), (1, ))
    assert_size_stride(arg296_1, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg300_1, (704, ), (1, ))
    assert_size_stride(arg301_1, (704, ), (1, ))
    assert_size_stride(arg302_1, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (128, ), (1, ))
    assert_size_stride(arg305_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg306_1, (736, ), (1, ))
    assert_size_stride(arg307_1, (736, ), (1, ))
    assert_size_stride(arg308_1, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (128, ), (1, ))
    assert_size_stride(arg317_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg318_1, (800, ), (1, ))
    assert_size_stride(arg319_1, (800, ), (1, ))
    assert_size_stride(arg320_1, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg321_1, (128, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg324_1, (832, ), (1, ))
    assert_size_stride(arg325_1, (832, ), (1, ))
    assert_size_stride(arg326_1, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg330_1, (864, ), (1, ))
    assert_size_stride(arg331_1, (864, ), (1, ))
    assert_size_stride(arg332_1, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg336_1, (896, ), (1, ))
    assert_size_stride(arg337_1, (896, ), (1, ))
    assert_size_stride(arg338_1, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg342_1, (928, ), (1, ))
    assert_size_stride(arg343_1, (928, ), (1, ))
    assert_size_stride(arg344_1, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg348_1, (960, ), (1, ))
    assert_size_stride(arg349_1, (960, ), (1, ))
    assert_size_stride(arg350_1, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg351_1, (128, ), (1, ))
    assert_size_stride(arg352_1, (128, ), (1, ))
    assert_size_stride(arg353_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg354_1, (992, ), (1, ))
    assert_size_stride(arg355_1, (992, ), (1, ))
    assert_size_stride(arg356_1, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg363_1, (1000, ), (1, ))
    assert_size_stride(arg364_1, (64, ), (1, ))
    assert_size_stride(arg365_1, (64, ), (1, ))
    assert_size_stride(arg366_1, (), ())
    assert_size_stride(arg367_1, (64, ), (1, ))
    assert_size_stride(arg368_1, (64, ), (1, ))
    assert_size_stride(arg369_1, (), ())
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (), ())
    assert_size_stride(arg373_1, (96, ), (1, ))
    assert_size_stride(arg374_1, (96, ), (1, ))
    assert_size_stride(arg375_1, (), ())
    assert_size_stride(arg376_1, (128, ), (1, ))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (), ())
    assert_size_stride(arg379_1, (128, ), (1, ))
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (), ())
    assert_size_stride(arg382_1, (128, ), (1, ))
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (), ())
    assert_size_stride(arg385_1, (160, ), (1, ))
    assert_size_stride(arg386_1, (160, ), (1, ))
    assert_size_stride(arg387_1, (), ())
    assert_size_stride(arg388_1, (128, ), (1, ))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (), ())
    assert_size_stride(arg391_1, (192, ), (1, ))
    assert_size_stride(arg392_1, (192, ), (1, ))
    assert_size_stride(arg393_1, (), ())
    assert_size_stride(arg394_1, (128, ), (1, ))
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (), ())
    assert_size_stride(arg397_1, (224, ), (1, ))
    assert_size_stride(arg398_1, (224, ), (1, ))
    assert_size_stride(arg399_1, (), ())
    assert_size_stride(arg400_1, (128, ), (1, ))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (), ())
    assert_size_stride(arg403_1, (256, ), (1, ))
    assert_size_stride(arg404_1, (256, ), (1, ))
    assert_size_stride(arg405_1, (), ())
    assert_size_stride(arg406_1, (128, ), (1, ))
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (), ())
    assert_size_stride(arg409_1, (128, ), (1, ))
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (), ())
    assert_size_stride(arg412_1, (160, ), (1, ))
    assert_size_stride(arg413_1, (160, ), (1, ))
    assert_size_stride(arg414_1, (), ())
    assert_size_stride(arg415_1, (128, ), (1, ))
    assert_size_stride(arg416_1, (128, ), (1, ))
    assert_size_stride(arg417_1, (), ())
    assert_size_stride(arg418_1, (192, ), (1, ))
    assert_size_stride(arg419_1, (192, ), (1, ))
    assert_size_stride(arg420_1, (), ())
    assert_size_stride(arg421_1, (128, ), (1, ))
    assert_size_stride(arg422_1, (128, ), (1, ))
    assert_size_stride(arg423_1, (), ())
    assert_size_stride(arg424_1, (224, ), (1, ))
    assert_size_stride(arg425_1, (224, ), (1, ))
    assert_size_stride(arg426_1, (), ())
    assert_size_stride(arg427_1, (128, ), (1, ))
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (), ())
    assert_size_stride(arg430_1, (256, ), (1, ))
    assert_size_stride(arg431_1, (256, ), (1, ))
    assert_size_stride(arg432_1, (), ())
    assert_size_stride(arg433_1, (128, ), (1, ))
    assert_size_stride(arg434_1, (128, ), (1, ))
    assert_size_stride(arg435_1, (), ())
    assert_size_stride(arg436_1, (288, ), (1, ))
    assert_size_stride(arg437_1, (288, ), (1, ))
    assert_size_stride(arg438_1, (), ())
    assert_size_stride(arg439_1, (128, ), (1, ))
    assert_size_stride(arg440_1, (128, ), (1, ))
    assert_size_stride(arg441_1, (), ())
    assert_size_stride(arg442_1, (320, ), (1, ))
    assert_size_stride(arg443_1, (320, ), (1, ))
    assert_size_stride(arg444_1, (), ())
    assert_size_stride(arg445_1, (128, ), (1, ))
    assert_size_stride(arg446_1, (128, ), (1, ))
    assert_size_stride(arg447_1, (), ())
    assert_size_stride(arg448_1, (352, ), (1, ))
    assert_size_stride(arg449_1, (352, ), (1, ))
    assert_size_stride(arg450_1, (), ())
    assert_size_stride(arg451_1, (128, ), (1, ))
    assert_size_stride(arg452_1, (128, ), (1, ))
    assert_size_stride(arg453_1, (), ())
    assert_size_stride(arg454_1, (384, ), (1, ))
    assert_size_stride(arg455_1, (384, ), (1, ))
    assert_size_stride(arg456_1, (), ())
    assert_size_stride(arg457_1, (128, ), (1, ))
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (), ())
    assert_size_stride(arg460_1, (416, ), (1, ))
    assert_size_stride(arg461_1, (416, ), (1, ))
    assert_size_stride(arg462_1, (), ())
    assert_size_stride(arg463_1, (128, ), (1, ))
    assert_size_stride(arg464_1, (128, ), (1, ))
    assert_size_stride(arg465_1, (), ())
    assert_size_stride(arg466_1, (448, ), (1, ))
    assert_size_stride(arg467_1, (448, ), (1, ))
    assert_size_stride(arg468_1, (), ())
    assert_size_stride(arg469_1, (128, ), (1, ))
    assert_size_stride(arg470_1, (128, ), (1, ))
    assert_size_stride(arg471_1, (), ())
    assert_size_stride(arg472_1, (480, ), (1, ))
    assert_size_stride(arg473_1, (480, ), (1, ))
    assert_size_stride(arg474_1, (), ())
    assert_size_stride(arg475_1, (128, ), (1, ))
    assert_size_stride(arg476_1, (128, ), (1, ))
    assert_size_stride(arg477_1, (), ())
    assert_size_stride(arg478_1, (512, ), (1, ))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (), ())
    assert_size_stride(arg481_1, (256, ), (1, ))
    assert_size_stride(arg482_1, (256, ), (1, ))
    assert_size_stride(arg483_1, (), ())
    assert_size_stride(arg484_1, (128, ), (1, ))
    assert_size_stride(arg485_1, (128, ), (1, ))
    assert_size_stride(arg486_1, (), ())
    assert_size_stride(arg487_1, (288, ), (1, ))
    assert_size_stride(arg488_1, (288, ), (1, ))
    assert_size_stride(arg489_1, (), ())
    assert_size_stride(arg490_1, (128, ), (1, ))
    assert_size_stride(arg491_1, (128, ), (1, ))
    assert_size_stride(arg492_1, (), ())
    assert_size_stride(arg493_1, (320, ), (1, ))
    assert_size_stride(arg494_1, (320, ), (1, ))
    assert_size_stride(arg495_1, (), ())
    assert_size_stride(arg496_1, (128, ), (1, ))
    assert_size_stride(arg497_1, (128, ), (1, ))
    assert_size_stride(arg498_1, (), ())
    assert_size_stride(arg499_1, (352, ), (1, ))
    assert_size_stride(arg500_1, (352, ), (1, ))
    assert_size_stride(arg501_1, (), ())
    assert_size_stride(arg502_1, (128, ), (1, ))
    assert_size_stride(arg503_1, (128, ), (1, ))
    assert_size_stride(arg504_1, (), ())
    assert_size_stride(arg505_1, (384, ), (1, ))
    assert_size_stride(arg506_1, (384, ), (1, ))
    assert_size_stride(arg507_1, (), ())
    assert_size_stride(arg508_1, (128, ), (1, ))
    assert_size_stride(arg509_1, (128, ), (1, ))
    assert_size_stride(arg510_1, (), ())
    assert_size_stride(arg511_1, (416, ), (1, ))
    assert_size_stride(arg512_1, (416, ), (1, ))
    assert_size_stride(arg513_1, (), ())
    assert_size_stride(arg514_1, (128, ), (1, ))
    assert_size_stride(arg515_1, (128, ), (1, ))
    assert_size_stride(arg516_1, (), ())
    assert_size_stride(arg517_1, (448, ), (1, ))
    assert_size_stride(arg518_1, (448, ), (1, ))
    assert_size_stride(arg519_1, (), ())
    assert_size_stride(arg520_1, (128, ), (1, ))
    assert_size_stride(arg521_1, (128, ), (1, ))
    assert_size_stride(arg522_1, (), ())
    assert_size_stride(arg523_1, (480, ), (1, ))
    assert_size_stride(arg524_1, (480, ), (1, ))
    assert_size_stride(arg525_1, (), ())
    assert_size_stride(arg526_1, (128, ), (1, ))
    assert_size_stride(arg527_1, (128, ), (1, ))
    assert_size_stride(arg528_1, (), ())
    assert_size_stride(arg529_1, (512, ), (1, ))
    assert_size_stride(arg530_1, (512, ), (1, ))
    assert_size_stride(arg531_1, (), ())
    assert_size_stride(arg532_1, (128, ), (1, ))
    assert_size_stride(arg533_1, (128, ), (1, ))
    assert_size_stride(arg534_1, (), ())
    assert_size_stride(arg535_1, (544, ), (1, ))
    assert_size_stride(arg536_1, (544, ), (1, ))
    assert_size_stride(arg537_1, (), ())
    assert_size_stride(arg538_1, (128, ), (1, ))
    assert_size_stride(arg539_1, (128, ), (1, ))
    assert_size_stride(arg540_1, (), ())
    assert_size_stride(arg541_1, (576, ), (1, ))
    assert_size_stride(arg542_1, (576, ), (1, ))
    assert_size_stride(arg543_1, (), ())
    assert_size_stride(arg544_1, (128, ), (1, ))
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (), ())
    assert_size_stride(arg547_1, (608, ), (1, ))
    assert_size_stride(arg548_1, (608, ), (1, ))
    assert_size_stride(arg549_1, (), ())
    assert_size_stride(arg550_1, (128, ), (1, ))
    assert_size_stride(arg551_1, (128, ), (1, ))
    assert_size_stride(arg552_1, (), ())
    assert_size_stride(arg553_1, (640, ), (1, ))
    assert_size_stride(arg554_1, (640, ), (1, ))
    assert_size_stride(arg555_1, (), ())
    assert_size_stride(arg556_1, (128, ), (1, ))
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (), ())
    assert_size_stride(arg559_1, (672, ), (1, ))
    assert_size_stride(arg560_1, (672, ), (1, ))
    assert_size_stride(arg561_1, (), ())
    assert_size_stride(arg562_1, (128, ), (1, ))
    assert_size_stride(arg563_1, (128, ), (1, ))
    assert_size_stride(arg564_1, (), ())
    assert_size_stride(arg565_1, (704, ), (1, ))
    assert_size_stride(arg566_1, (704, ), (1, ))
    assert_size_stride(arg567_1, (), ())
    assert_size_stride(arg568_1, (128, ), (1, ))
    assert_size_stride(arg569_1, (128, ), (1, ))
    assert_size_stride(arg570_1, (), ())
    assert_size_stride(arg571_1, (736, ), (1, ))
    assert_size_stride(arg572_1, (736, ), (1, ))
    assert_size_stride(arg573_1, (), ())
    assert_size_stride(arg574_1, (128, ), (1, ))
    assert_size_stride(arg575_1, (128, ), (1, ))
    assert_size_stride(arg576_1, (), ())
    assert_size_stride(arg577_1, (768, ), (1, ))
    assert_size_stride(arg578_1, (768, ), (1, ))
    assert_size_stride(arg579_1, (), ())
    assert_size_stride(arg580_1, (128, ), (1, ))
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (), ())
    assert_size_stride(arg583_1, (800, ), (1, ))
    assert_size_stride(arg584_1, (800, ), (1, ))
    assert_size_stride(arg585_1, (), ())
    assert_size_stride(arg586_1, (128, ), (1, ))
    assert_size_stride(arg587_1, (128, ), (1, ))
    assert_size_stride(arg588_1, (), ())
    assert_size_stride(arg589_1, (832, ), (1, ))
    assert_size_stride(arg590_1, (832, ), (1, ))
    assert_size_stride(arg591_1, (), ())
    assert_size_stride(arg592_1, (128, ), (1, ))
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (), ())
    assert_size_stride(arg595_1, (864, ), (1, ))
    assert_size_stride(arg596_1, (864, ), (1, ))
    assert_size_stride(arg597_1, (), ())
    assert_size_stride(arg598_1, (128, ), (1, ))
    assert_size_stride(arg599_1, (128, ), (1, ))
    assert_size_stride(arg600_1, (), ())
    assert_size_stride(arg601_1, (896, ), (1, ))
    assert_size_stride(arg602_1, (896, ), (1, ))
    assert_size_stride(arg603_1, (), ())
    assert_size_stride(arg604_1, (128, ), (1, ))
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (), ())
    assert_size_stride(arg607_1, (928, ), (1, ))
    assert_size_stride(arg608_1, (928, ), (1, ))
    assert_size_stride(arg609_1, (), ())
    assert_size_stride(arg610_1, (128, ), (1, ))
    assert_size_stride(arg611_1, (128, ), (1, ))
    assert_size_stride(arg612_1, (), ())
    assert_size_stride(arg613_1, (960, ), (1, ))
    assert_size_stride(arg614_1, (960, ), (1, ))
    assert_size_stride(arg615_1, (), ())
    assert_size_stride(arg616_1, (128, ), (1, ))
    assert_size_stride(arg617_1, (128, ), (1, ))
    assert_size_stride(arg618_1, (), ())
    assert_size_stride(arg619_1, (992, ), (1, ))
    assert_size_stride(arg620_1, (992, ), (1, ))
    assert_size_stride(arg621_1, (), ())
    assert_size_stride(arg622_1, (128, ), (1, ))
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (), ())
    assert_size_stride(arg625_1, (1024, ), (1, ))
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (), ())
    assert_size_stride(arg628_1, (512, ), (1, ))
    assert_size_stride(arg629_1, (512, ), (1, ))
    assert_size_stride(arg630_1, (), ())
    assert_size_stride(arg631_1, (128, ), (1, ))
    assert_size_stride(arg632_1, (128, ), (1, ))
    assert_size_stride(arg633_1, (), ())
    assert_size_stride(arg634_1, (544, ), (1, ))
    assert_size_stride(arg635_1, (544, ), (1, ))
    assert_size_stride(arg636_1, (), ())
    assert_size_stride(arg637_1, (128, ), (1, ))
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (), ())
    assert_size_stride(arg640_1, (576, ), (1, ))
    assert_size_stride(arg641_1, (576, ), (1, ))
    assert_size_stride(arg642_1, (), ())
    assert_size_stride(arg643_1, (128, ), (1, ))
    assert_size_stride(arg644_1, (128, ), (1, ))
    assert_size_stride(arg645_1, (), ())
    assert_size_stride(arg646_1, (608, ), (1, ))
    assert_size_stride(arg647_1, (608, ), (1, ))
    assert_size_stride(arg648_1, (), ())
    assert_size_stride(arg649_1, (128, ), (1, ))
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (), ())
    assert_size_stride(arg652_1, (640, ), (1, ))
    assert_size_stride(arg653_1, (640, ), (1, ))
    assert_size_stride(arg654_1, (), ())
    assert_size_stride(arg655_1, (128, ), (1, ))
    assert_size_stride(arg656_1, (128, ), (1, ))
    assert_size_stride(arg657_1, (), ())
    assert_size_stride(arg658_1, (672, ), (1, ))
    assert_size_stride(arg659_1, (672, ), (1, ))
    assert_size_stride(arg660_1, (), ())
    assert_size_stride(arg661_1, (128, ), (1, ))
    assert_size_stride(arg662_1, (128, ), (1, ))
    assert_size_stride(arg663_1, (), ())
    assert_size_stride(arg664_1, (704, ), (1, ))
    assert_size_stride(arg665_1, (704, ), (1, ))
    assert_size_stride(arg666_1, (), ())
    assert_size_stride(arg667_1, (128, ), (1, ))
    assert_size_stride(arg668_1, (128, ), (1, ))
    assert_size_stride(arg669_1, (), ())
    assert_size_stride(arg670_1, (736, ), (1, ))
    assert_size_stride(arg671_1, (736, ), (1, ))
    assert_size_stride(arg672_1, (), ())
    assert_size_stride(arg673_1, (128, ), (1, ))
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (), ())
    assert_size_stride(arg676_1, (768, ), (1, ))
    assert_size_stride(arg677_1, (768, ), (1, ))
    assert_size_stride(arg678_1, (), ())
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (128, ), (1, ))
    assert_size_stride(arg681_1, (), ())
    assert_size_stride(arg682_1, (800, ), (1, ))
    assert_size_stride(arg683_1, (800, ), (1, ))
    assert_size_stride(arg684_1, (), ())
    assert_size_stride(arg685_1, (128, ), (1, ))
    assert_size_stride(arg686_1, (128, ), (1, ))
    assert_size_stride(arg687_1, (), ())
    assert_size_stride(arg688_1, (832, ), (1, ))
    assert_size_stride(arg689_1, (832, ), (1, ))
    assert_size_stride(arg690_1, (), ())
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (128, ), (1, ))
    assert_size_stride(arg693_1, (), ())
    assert_size_stride(arg694_1, (864, ), (1, ))
    assert_size_stride(arg695_1, (864, ), (1, ))
    assert_size_stride(arg696_1, (), ())
    assert_size_stride(arg697_1, (128, ), (1, ))
    assert_size_stride(arg698_1, (128, ), (1, ))
    assert_size_stride(arg699_1, (), ())
    assert_size_stride(arg700_1, (896, ), (1, ))
    assert_size_stride(arg701_1, (896, ), (1, ))
    assert_size_stride(arg702_1, (), ())
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (128, ), (1, ))
    assert_size_stride(arg705_1, (), ())
    assert_size_stride(arg706_1, (928, ), (1, ))
    assert_size_stride(arg707_1, (928, ), (1, ))
    assert_size_stride(arg708_1, (), ())
    assert_size_stride(arg709_1, (128, ), (1, ))
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (), ())
    assert_size_stride(arg712_1, (960, ), (1, ))
    assert_size_stride(arg713_1, (960, ), (1, ))
    assert_size_stride(arg714_1, (), ())
    assert_size_stride(arg715_1, (128, ), (1, ))
    assert_size_stride(arg716_1, (128, ), (1, ))
    assert_size_stride(arg717_1, (), ())
    assert_size_stride(arg718_1, (992, ), (1, ))
    assert_size_stride(arg719_1, (992, ), (1, ))
    assert_size_stride(arg720_1, (), ())
    assert_size_stride(arg721_1, (128, ), (1, ))
    assert_size_stride(arg722_1, (128, ), (1, ))
    assert_size_stride(arg723_1, (), ())
    assert_size_stride(arg724_1, (1024, ), (1, ))
    assert_size_stride(arg725_1, (1024, ), (1, ))
    assert_size_stride(arg726_1, (), ())
    assert_size_stride(arg727_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg727_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg727_1
    # Source Nodes: [l__mod___features_conv0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 64, 112, 112), (802816, 1, 7168, 64))
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf29 = empty_strided((4, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    buf4 = reinterpret_tensor(buf29, (4, 64, 56, 56), (602112, 1, 10752, 192), 0)  # alias
    buf5 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    buf35 = reinterpret_tensor(buf41, (4, 64, 56, 56), (702464, 1, 12544, 224), 0)  # alias
    buf54 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf54, (4, 64, 56, 56), (802816, 1, 14336, 256), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg1_1
    del arg2_1
    del arg364_1
    del arg365_1
    del arg367_1
    del arg368_1
    del arg3_1
    del arg4_1
    del buf3
    # Source Nodes: [bottleneck_output, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg5_1
    buf7 = buf6; del buf6  # reuse
    buf8 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf7.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg370_1
    del arg371_1
    del arg6_1
    del arg7_1
    del arg8_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2, new_features], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (4, 32, 56, 56), (100352, 1, 1792, 32))
    del buf7
    buf10 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_3(c_void_p(buf4.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg10_1
    del arg373_1
    del arg374_1
    del arg9_1
    # Source Nodes: [bottleneck_output_2, cat_122, l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf10, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg11_1
    buf12 = buf11; del buf11  # reuse
    buf13 = buf8; del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg12_1
    del arg13_1
    del arg14_1
    del arg376_1
    del arg377_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer2_norm2, l__mod___features_denseblock1_denselayer2_relu2, new_features_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf15 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_5(c_void_p(buf4.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg15_1
    del arg16_1
    del arg379_1
    del arg380_1
    # Source Nodes: [bottleneck_output_4, cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf16 = extern_kernels.convolution(buf15, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg17_1
    del buf15
    buf17 = buf16; del buf16  # reuse
    buf18 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_6(c_void_p(buf17.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg18_1
    del arg19_1
    del arg20_1
    del arg382_1
    del arg383_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer3_norm2, l__mod___features_denseblock1_denselayer3_relu2, new_features_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (4, 32, 56, 56), (100352, 1, 1792, 32))
    del buf17
    buf20 = empty_strided((4, 160, 56, 56), (501760, 1, 8960, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_7(c_void_p(buf4.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg21_1
    del arg22_1
    del arg385_1
    del arg386_1
    # Source Nodes: [bottleneck_output_6, cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf21 = extern_kernels.convolution(buf20, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg23_1
    del buf20
    buf22 = buf21; del buf21  # reuse
    buf23 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8(c_void_p(buf22.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg24_1
    del arg25_1
    del arg26_1
    del arg388_1
    del arg389_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer4_norm2, l__mod___features_denseblock1_denselayer4_relu2, new_features_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (4, 32, 56, 56), (100352, 1, 1792, 32))
    del buf22
    buf25 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 1, 10752, 192), 64)  # alias
    buf36 = reinterpret_tensor(buf41, (4, 32, 56, 56), (702464, 1, 12544, 224), 64)  # alias
    buf48 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 64)  # alias
    buf26 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 1, 10752, 192), 96)  # alias
    buf37 = reinterpret_tensor(buf41, (4, 32, 56, 56), (702464, 1, 12544, 224), 96)  # alias
    buf49 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 96)  # alias
    buf27 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 1, 10752, 192), 128)  # alias
    buf38 = reinterpret_tensor(buf41, (4, 32, 56, 56), (702464, 1, 12544, 224), 128)  # alias
    buf50 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 128)  # alias
    buf28 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 1, 10752, 192), 160)  # alias
    buf39 = reinterpret_tensor(buf41, (4, 32, 56, 56), (702464, 1, 12544, 224), 160)  # alias
    buf51 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 160)  # alias
    buf30 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_9(c_void_p(buf30.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg27_1
    del arg28_1
    del arg391_1
    del arg392_1
    del buf14
    del buf19
    del buf24
    del buf25
    del buf26
    del buf27
    del buf28
    del buf4
    del buf9
    # Source Nodes: [bottleneck_output_8, l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf31 = extern_kernels.convolution(buf30, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg29_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    buf33 = buf23; del buf23  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_10(c_void_p(buf32.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg30_1
    del arg31_1
    del arg32_1
    del arg394_1
    del arg395_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer5_norm2, l__mod___features_denseblock1_denselayer5_relu2, new_features_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (4, 32, 56, 56), (100352, 1, 1792, 32))
    del buf32
    buf40 = reinterpret_tensor(buf41, (4, 32, 56, 56), (702464, 1, 12544, 224), 192)  # alias
    buf52 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 192)  # alias
    buf42 = buf41; del buf41  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_11(c_void_p(buf42.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg33_1
    del arg34_1
    del arg397_1
    del arg398_1
    del buf34
    del buf35
    del buf36
    del buf37
    del buf38
    del buf39
    del buf40
    # Source Nodes: [bottleneck_output_10, l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf43 = extern_kernels.convolution(buf42, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg35_1
    del buf42
    buf44 = buf43; del buf43  # reuse
    buf45 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12(c_void_p(buf44.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf45.data_ptr()))
    del arg36_1
    del arg37_1
    del arg38_1
    del arg400_1
    del arg401_1
    # Source Nodes: [l__mod___features_denseblock1_denselayer6_norm2, l__mod___features_denseblock1_denselayer6_relu2, new_features_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (4, 32, 56, 56), (100352, 1, 1792, 32))
    del buf44
    buf53 = reinterpret_tensor(buf54, (4, 32, 56, 56), (802816, 1, 14336, 256), 224)  # alias
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_13(c_void_p(buf55.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg39_1
    del arg403_1
    del arg404_1
    del arg40_1
    del buf47
    del buf48
    del buf49
    del buf50
    del buf51
    del buf52
    del buf53
    # Source Nodes: [l__mod___features_transition1_conv, l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf56 = extern_kernels.convolution(buf55, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (4, 128, 56, 56), (401408, 1, 7168, 128))
    del arg41_1
    del buf55
    buf57 = reinterpret_tensor(buf46, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf46  # reuse
    buf85 = reinterpret_tensor(buf5, (4, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf5  # reuse
    buf80 = reinterpret_tensor(buf85, (4, 128, 28, 28), (200704, 1, 7168, 256), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_14(c_void_p(buf56.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg406_1
    del arg407_1
    del arg42_1
    del arg43_1
    # Source Nodes: [bottleneck_output_12, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf57, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg44_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    buf60 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15(c_void_p(buf59.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg409_1
    del arg410_1
    del arg45_1
    del arg46_1
    del arg47_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2, new_features_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf61 = extern_kernels.convolution(buf59, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf61, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf59
    buf62 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_16(c_void_p(buf63.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg412_1
    del arg413_1
    del arg48_1
    del arg49_1
    # Source Nodes: [bottleneck_output_14, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
    buf64 = extern_kernels.convolution(buf63, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg50_1
    buf65 = buf64; del buf64  # reuse
    buf66 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf65.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg415_1
    del arg416_1
    del arg51_1
    del arg52_1
    del arg53_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer2_norm2, l__mod___features_denseblock2_denselayer2_relu2, new_features_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf67 = extern_kernels.convolution(buf65, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf65
    buf68 = reinterpret_tensor(buf0, (4, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf0  # reuse
    buf69 = buf68; del buf68  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_18(c_void_p(buf69.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg418_1
    del arg419_1
    del arg54_1
    del arg55_1
    # Source Nodes: [bottleneck_output_16, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf70 = extern_kernels.convolution(buf69, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg56_1
    buf71 = buf70; del buf70  # reuse
    buf72 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19(c_void_p(buf71.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg421_1
    del arg422_1
    del arg57_1
    del arg58_1
    del arg59_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer3_norm2, l__mod___features_denseblock2_denselayer3_relu2, new_features_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf73 = extern_kernels.convolution(buf71, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf71
    buf74 = empty_strided((4, 224, 28, 28), (175616, 1, 6272, 224), device='cpu', dtype=torch.float32)
    buf75 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_20(c_void_p(buf75.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg424_1
    del arg425_1
    del arg60_1
    del arg61_1
    # Source Nodes: [bottleneck_output_18, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf76 = extern_kernels.convolution(buf75, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg62_1
    buf77 = buf76; del buf76  # reuse
    buf78 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21(c_void_p(buf77.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg427_1
    del arg428_1
    del arg63_1
    del arg64_1
    del arg65_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer4_norm2, l__mod___features_denseblock2_denselayer4_relu2, new_features_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf79 = extern_kernels.convolution(buf77, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf77
    buf81 = reinterpret_tensor(buf85, (4, 32, 28, 28), (200704, 1, 7168, 256), 128)  # alias
    buf97 = empty_strided((4, 288, 28, 28), (225792, 1, 8064, 288), device='cpu', dtype=torch.float32)
    buf92 = reinterpret_tensor(buf97, (4, 32, 28, 28), (225792, 1, 8064, 288), 128)  # alias
    buf110 = empty_strided((4, 320, 28, 28), (250880, 1, 8960, 320), device='cpu', dtype=torch.float32)
    buf104 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 128)  # alias
    buf124 = empty_strided((4, 352, 28, 28), (275968, 1, 9856, 352), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 128)  # alias
    buf139 = reinterpret_tensor(buf10, (4, 384, 28, 28), (301056, 1, 10752, 384), 0); del buf10  # reuse
    buf131 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 128)  # alias
    buf82 = reinterpret_tensor(buf85, (4, 32, 28, 28), (200704, 1, 7168, 256), 160)  # alias
    buf93 = reinterpret_tensor(buf97, (4, 32, 28, 28), (225792, 1, 8064, 288), 160)  # alias
    buf105 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 160)  # alias
    buf118 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 160)  # alias
    buf132 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 160)  # alias
    buf83 = reinterpret_tensor(buf85, (4, 32, 28, 28), (200704, 1, 7168, 256), 192)  # alias
    buf94 = reinterpret_tensor(buf97, (4, 32, 28, 28), (225792, 1, 8064, 288), 192)  # alias
    buf106 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 192)  # alias
    buf119 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 192)  # alias
    buf133 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 192)  # alias
    buf84 = reinterpret_tensor(buf85, (4, 32, 28, 28), (200704, 1, 7168, 256), 224)  # alias
    buf95 = reinterpret_tensor(buf97, (4, 32, 28, 28), (225792, 1, 8064, 288), 224)  # alias
    buf107 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 224)  # alias
    buf120 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 224)  # alias
    buf134 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 224)  # alias
    buf86 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_22(c_void_p(buf61.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf86.data_ptr()))
    del arg430_1
    del arg431_1
    del arg66_1
    del arg67_1
    del buf81
    del buf82
    del buf83
    del buf84
    # Source Nodes: [bottleneck_output_20, l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf87 = extern_kernels.convolution(buf86, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg68_1
    del buf86
    buf88 = buf87; del buf87  # reuse
    buf89 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23(c_void_p(buf88.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg433_1
    del arg434_1
    del arg69_1
    del arg70_1
    del arg71_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer5_norm2, l__mod___features_denseblock2_denselayer5_relu2, new_features_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf88
    buf91 = reinterpret_tensor(buf97, (4, 128, 28, 28), (225792, 1, 8064, 288), 0)  # alias
    buf103 = reinterpret_tensor(buf110, (4, 128, 28, 28), (250880, 1, 8960, 320), 0)  # alias
    buf116 = reinterpret_tensor(buf124, (4, 128, 28, 28), (275968, 1, 9856, 352), 0)  # alias
    buf130 = reinterpret_tensor(buf139, (4, 128, 28, 28), (301056, 1, 10752, 384), 0)  # alias
    buf155 = empty_strided((4, 416, 28, 28), (326144, 1, 11648, 416), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf155, (4, 128, 28, 28), (326144, 1, 11648, 416), 0)  # alias
    buf96 = reinterpret_tensor(buf97, (4, 32, 28, 28), (225792, 1, 8064, 288), 256)  # alias
    buf108 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 256)  # alias
    buf121 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 256)  # alias
    buf135 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 256)  # alias
    buf150 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 256)  # alias
    buf98 = buf97; del buf97  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_24(c_void_p(buf98.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf150.data_ptr()))
    del arg436_1
    del arg437_1
    del arg72_1
    del arg73_1
    del buf91
    del buf92
    del buf93
    del buf94
    del buf95
    del buf96
    # Source Nodes: [bottleneck_output_22, l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf99 = extern_kernels.convolution(buf98, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg74_1
    del buf98
    buf100 = buf99; del buf99  # reuse
    buf101 = buf89; del buf89  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25(c_void_p(buf100.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg439_1
    del arg440_1
    del arg75_1
    del arg76_1
    del arg77_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer6_norm2, l__mod___features_denseblock2_denselayer6_relu2, new_features_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf100
    buf109 = reinterpret_tensor(buf110, (4, 32, 28, 28), (250880, 1, 8960, 320), 288)  # alias
    buf122 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 288)  # alias
    buf136 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 288)  # alias
    buf151 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 288)  # alias
    buf172 = empty_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 288)  # alias
    buf111 = buf110; del buf110  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_26(c_void_p(buf111.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg442_1
    del arg443_1
    del arg78_1
    del arg79_1
    del buf103
    del buf104
    del buf105
    del buf106
    del buf107
    del buf108
    del buf109
    # Source Nodes: [bottleneck_output_24, l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf111, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf112, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg80_1
    del buf111
    buf113 = buf112; del buf112  # reuse
    buf114 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27(c_void_p(buf113.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg445_1
    del arg446_1
    del arg81_1
    del arg82_1
    del arg83_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer7_norm2, l__mod___features_denseblock2_denselayer7_relu2, new_features_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf115 = extern_kernels.convolution(buf113, buf114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf115, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf113
    buf123 = reinterpret_tensor(buf124, (4, 32, 28, 28), (275968, 1, 9856, 352), 320)  # alias
    buf137 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 320)  # alias
    buf152 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 320)  # alias
    buf168 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 320)  # alias
    buf190 = empty_strided((4, 480, 28, 28), (376320, 1, 13440, 480), device='cpu', dtype=torch.float32)
    buf185 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 320)  # alias
    buf125 = buf124; del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_28(c_void_p(buf125.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf185.data_ptr()))
    del arg448_1
    del arg449_1
    del arg84_1
    del arg85_1
    del buf116
    del buf117
    del buf118
    del buf119
    del buf120
    del buf121
    del buf122
    del buf123
    # Source Nodes: [bottleneck_output_26, l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf125, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg86_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    buf128 = buf114; del buf114  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29(c_void_p(buf127.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg451_1
    del arg452_1
    del arg87_1
    del arg88_1
    del arg89_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer8_norm2, l__mod___features_denseblock2_denselayer8_relu2, new_features_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf129, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf127
    buf138 = reinterpret_tensor(buf139, (4, 32, 28, 28), (301056, 1, 10752, 384), 352)  # alias
    buf153 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 352)  # alias
    buf169 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 352)  # alias
    buf186 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 352)  # alias
    buf140 = buf139; del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_30(c_void_p(buf140.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg454_1
    del arg455_1
    del arg90_1
    del arg91_1
    del buf130
    del buf131
    del buf132
    del buf133
    del buf134
    del buf135
    del buf136
    del buf137
    del buf138
    # Source Nodes: [bottleneck_output_28, l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf141 = extern_kernels.convolution(buf140, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf141, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg92_1
    del buf140
    buf142 = buf141; del buf141  # reuse
    buf143 = buf128; del buf128  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_31(c_void_p(buf142.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf143.data_ptr()))
    del arg457_1
    del arg458_1
    del arg93_1
    del arg94_1
    del arg95_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer9_norm2, l__mod___features_denseblock2_denselayer9_relu2, new_features_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf144 = extern_kernels.convolution(buf142, buf143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf144, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf142
    buf146 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 128)  # alias
    buf162 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 128)  # alias
    buf179 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 128)  # alias
    buf209 = reinterpret_tensor(buf56, (4, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf56  # reuse
    buf197 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 128)  # alias
    buf147 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 160)  # alias
    buf163 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 160)  # alias
    buf180 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 160)  # alias
    buf198 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 160)  # alias
    buf148 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 192)  # alias
    buf164 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 192)  # alias
    buf181 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 192)  # alias
    buf199 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 192)  # alias
    buf149 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 224)  # alias
    buf165 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 224)  # alias
    buf182 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 224)  # alias
    buf200 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 224)  # alias
    buf154 = reinterpret_tensor(buf155, (4, 32, 28, 28), (326144, 1, 11648, 416), 384)  # alias
    buf170 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 384)  # alias
    buf187 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 384)  # alias
    buf205 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 384)  # alias
    buf156 = buf155; del buf155  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_32(c_void_p(buf156.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(arg460_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg460_1
    del arg461_1
    del arg96_1
    del arg97_1
    del buf144
    del buf145
    del buf146
    del buf147
    del buf148
    del buf149
    del buf150
    del buf151
    del buf152
    del buf153
    del buf154
    del buf61
    del buf67
    del buf73
    del buf79
    # Source Nodes: [bottleneck_output_30, l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf157 = extern_kernels.convolution(buf156, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf157, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg98_1
    del buf156
    buf158 = buf157; del buf157  # reuse
    buf159 = buf143; del buf143  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_33(c_void_p(buf158.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf159.data_ptr()))
    del arg100_1
    del arg101_1
    del arg463_1
    del arg464_1
    del arg99_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer10_norm2, l__mod___features_denseblock2_denselayer10_relu2, new_features_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf160, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf158
    buf161 = reinterpret_tensor(buf172, (4, 128, 28, 28), (351232, 1, 12544, 448), 0)  # alias
    buf178 = reinterpret_tensor(buf190, (4, 128, 28, 28), (376320, 1, 13440, 480), 0)  # alias
    buf196 = reinterpret_tensor(buf209, (4, 128, 28, 28), (401408, 1, 14336, 512), 0)  # alias
    buf166 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 256)  # alias
    buf183 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 256)  # alias
    buf201 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 256)  # alias
    buf171 = reinterpret_tensor(buf172, (4, 32, 28, 28), (351232, 1, 12544, 448), 416)  # alias
    buf188 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 416)  # alias
    buf206 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 416)  # alias
    buf173 = buf172; del buf172  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_34(c_void_p(buf173.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf206.data_ptr()))
    del arg102_1
    del arg103_1
    del arg466_1
    del arg467_1
    del buf160
    del buf161
    del buf162
    del buf163
    del buf164
    del buf165
    del buf166
    del buf167
    del buf168
    del buf169
    del buf170
    del buf171
    del buf80
    del buf85
    del buf90
    # Source Nodes: [bottleneck_output_32, l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf174 = extern_kernels.convolution(buf173, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf174, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg104_1
    del buf173
    buf175 = buf174; del buf174  # reuse
    buf176 = buf159; del buf159  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35(c_void_p(buf175.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf176.data_ptr()))
    del arg105_1
    del arg106_1
    del arg107_1
    del arg469_1
    del arg470_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer11_norm2, l__mod___features_denseblock2_denselayer11_relu2, new_features_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf177 = extern_kernels.convolution(buf175, buf176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf177, (4, 32, 28, 28), (25088, 1, 896, 32))
    del buf175
    buf184 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 288)  # alias
    buf202 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 288)  # alias
    buf189 = reinterpret_tensor(buf190, (4, 32, 28, 28), (376320, 1, 13440, 480), 448)  # alias
    buf207 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 448)  # alias
    buf191 = buf190; del buf190  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_36(c_void_p(buf191.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg108_1
    del arg109_1
    del arg472_1
    del arg473_1
    del buf102
    del buf177
    del buf178
    del buf179
    del buf180
    del buf181
    del buf182
    del buf183
    del buf184
    del buf185
    del buf186
    del buf187
    del buf188
    del buf189
    # Source Nodes: [bottleneck_output_34, l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf192 = extern_kernels.convolution(buf191, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf192, (4, 128, 28, 28), (100352, 1, 3584, 128))
    del arg110_1
    del buf191
    buf193 = buf192; del buf192  # reuse
    buf194 = buf176; del buf176  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37(c_void_p(buf193.data_ptr()), c_void_p(arg475_1.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg111_1
    del arg112_1
    del arg113_1
    del arg475_1
    del arg476_1
    # Source Nodes: [l__mod___features_denseblock2_denselayer12_norm2, l__mod___features_denseblock2_denselayer12_relu2, new_features_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf195 = extern_kernels.convolution(buf193, buf194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf195, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf203 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 320)  # alias
    buf204 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 352)  # alias
    buf208 = reinterpret_tensor(buf209, (4, 32, 28, 28), (401408, 1, 14336, 512), 480)  # alias
    buf210 = buf209; del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_38(c_void_p(buf210.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg478_1.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg114_1
    del arg115_1
    del arg478_1
    del arg479_1
    del buf115
    del buf129
    del buf195
    del buf196
    del buf197
    del buf198
    del buf199
    del buf200
    del buf201
    del buf202
    del buf203
    del buf204
    del buf205
    del buf206
    del buf207
    del buf208
    # Source Nodes: [l__mod___features_transition2_conv, l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf211 = extern_kernels.convolution(buf210, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf211, (4, 256, 28, 28), (200704, 1, 7168, 256))
    del arg116_1
    del buf210
    buf212 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    buf240 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf235 = reinterpret_tensor(buf240, (4, 256, 14, 14), (75264, 1, 5376, 384), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_39(c_void_p(buf211.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf235.data_ptr()))
    del arg117_1
    del arg118_1
    del arg481_1
    del arg482_1
    # Source Nodes: [bottleneck_output_36, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
    buf213 = extern_kernels.convolution(buf212, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf213, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg119_1
    buf214 = buf213; del buf213  # reuse
    buf215 = buf194; del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40(c_void_p(buf214.data_ptr()), c_void_p(arg484_1.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf215.data_ptr()))
    del arg120_1
    del arg121_1
    del arg122_1
    del arg484_1
    del arg485_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2, new_features_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf216 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf216, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf214
    buf217 = empty_strided((4, 288, 14, 14), (56448, 1, 4032, 288), device='cpu', dtype=torch.float32)
    buf218 = buf217; del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_41(c_void_p(buf218.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(arg487_1.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()))
    del arg123_1
    del arg124_1
    del arg487_1
    del arg488_1
    # Source Nodes: [bottleneck_output_38, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
    buf219 = extern_kernels.convolution(buf218, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf219, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg125_1
    del buf218
    buf220 = buf219; del buf219  # reuse
    buf221 = buf215; del buf215  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42(c_void_p(buf220.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf221.data_ptr()))
    del arg126_1
    del arg127_1
    del arg128_1
    del arg490_1
    del arg491_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer2_norm2, l__mod___features_denseblock3_denselayer2_relu2, new_features_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf222, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf220
    buf223 = empty_strided((4, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    buf224 = buf223; del buf223  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_43(c_void_p(buf224.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg493_1.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg493_1
    del arg494_1
    # Source Nodes: [bottleneck_output_40, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf225 = extern_kernels.convolution(buf224, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf225, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg131_1
    del buf224
    buf226 = buf225; del buf225  # reuse
    buf227 = buf221; del buf221  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44(c_void_p(buf226.data_ptr()), c_void_p(arg496_1.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg132_1
    del arg133_1
    del arg134_1
    del arg496_1
    del arg497_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer3_norm2, l__mod___features_denseblock3_denselayer3_relu2, new_features_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf226
    buf229 = empty_strided((4, 352, 14, 14), (68992, 1, 4928, 352), device='cpu', dtype=torch.float32)
    buf230 = buf229; del buf229  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_45(c_void_p(buf230.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()))
    del arg135_1
    del arg136_1
    del arg499_1
    del arg500_1
    # Source Nodes: [bottleneck_output_42, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf231 = extern_kernels.convolution(buf230, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf231, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg137_1
    del buf230
    buf232 = buf231; del buf231  # reuse
    buf233 = buf227; del buf227  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_46(c_void_p(buf232.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg138_1
    del arg139_1
    del arg140_1
    del arg502_1
    del arg503_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer4_norm2, l__mod___features_denseblock3_denselayer4_relu2, new_features_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf234, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf232
    buf236 = reinterpret_tensor(buf240, (4, 32, 14, 14), (75264, 1, 5376, 384), 256)  # alias
    buf252 = empty_strided((4, 416, 14, 14), (81536, 1, 5824, 416), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf252, (4, 32, 14, 14), (81536, 1, 5824, 416), 256)  # alias
    buf265 = empty_strided((4, 448, 14, 14), (87808, 1, 6272, 448), device='cpu', dtype=torch.float32)
    buf259 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 256)  # alias
    buf279 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf272 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 256)  # alias
    buf294 = reinterpret_tensor(buf193, (4, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf193  # reuse
    buf286 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 256)  # alias
    buf237 = reinterpret_tensor(buf240, (4, 32, 14, 14), (75264, 1, 5376, 384), 288)  # alias
    buf248 = reinterpret_tensor(buf252, (4, 32, 14, 14), (81536, 1, 5824, 416), 288)  # alias
    buf260 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 288)  # alias
    buf273 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 288)  # alias
    buf287 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 288)  # alias
    buf238 = reinterpret_tensor(buf240, (4, 32, 14, 14), (75264, 1, 5376, 384), 320)  # alias
    buf249 = reinterpret_tensor(buf252, (4, 32, 14, 14), (81536, 1, 5824, 416), 320)  # alias
    buf261 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 320)  # alias
    buf274 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 320)  # alias
    buf288 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 320)  # alias
    buf239 = reinterpret_tensor(buf240, (4, 32, 14, 14), (75264, 1, 5376, 384), 352)  # alias
    buf250 = reinterpret_tensor(buf252, (4, 32, 14, 14), (81536, 1, 5824, 416), 352)  # alias
    buf262 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 352)  # alias
    buf275 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 352)  # alias
    buf289 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 352)  # alias
    buf241 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_47(c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg141_1
    del arg142_1
    del arg505_1
    del arg506_1
    del buf236
    del buf237
    del buf238
    del buf239
    # Source Nodes: [bottleneck_output_44, l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf242 = extern_kernels.convolution(buf241, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg143_1
    del buf241
    buf243 = buf242; del buf242  # reuse
    buf244 = buf233; del buf233  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_48(c_void_p(buf243.data_ptr()), c_void_p(arg508_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf244.data_ptr()))
    del arg144_1
    del arg145_1
    del arg146_1
    del arg508_1
    del arg509_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer5_norm2, l__mod___features_denseblock3_denselayer5_relu2, new_features_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf245 = extern_kernels.convolution(buf243, buf244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf245, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf243
    buf246 = reinterpret_tensor(buf252, (4, 256, 14, 14), (81536, 1, 5824, 416), 0)  # alias
    buf258 = reinterpret_tensor(buf265, (4, 256, 14, 14), (87808, 1, 6272, 448), 0)  # alias
    buf271 = reinterpret_tensor(buf279, (4, 256, 14, 14), (94080, 1, 6720, 480), 0)  # alias
    buf285 = reinterpret_tensor(buf294, (4, 256, 14, 14), (100352, 1, 7168, 512), 0)  # alias
    buf310 = empty_strided((4, 544, 14, 14), (106624, 1, 7616, 544), device='cpu', dtype=torch.float32)
    buf300 = reinterpret_tensor(buf310, (4, 256, 14, 14), (106624, 1, 7616, 544), 0)  # alias
    buf251 = reinterpret_tensor(buf252, (4, 32, 14, 14), (81536, 1, 5824, 416), 384)  # alias
    buf263 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 384)  # alias
    buf276 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 384)  # alias
    buf290 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 384)  # alias
    buf305 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 384)  # alias
    buf253 = buf252; del buf252  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_49(c_void_p(buf253.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(arg511_1.data_ptr()), c_void_p(arg512_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf305.data_ptr()))
    del arg147_1
    del arg148_1
    del arg511_1
    del arg512_1
    del buf246
    del buf247
    del buf248
    del buf249
    del buf250
    del buf251
    # Source Nodes: [bottleneck_output_46, l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf254 = extern_kernels.convolution(buf253, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf254, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg149_1
    del buf253
    buf255 = buf254; del buf254  # reuse
    buf256 = buf244; del buf244  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50(c_void_p(buf255.data_ptr()), c_void_p(arg514_1.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf256.data_ptr()))
    del arg150_1
    del arg151_1
    del arg152_1
    del arg514_1
    del arg515_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer6_norm2, l__mod___features_denseblock3_denselayer6_relu2, new_features_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf257 = extern_kernels.convolution(buf255, buf256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf255
    buf264 = reinterpret_tensor(buf265, (4, 32, 14, 14), (87808, 1, 6272, 448), 416)  # alias
    buf277 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 416)  # alias
    buf291 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 416)  # alias
    buf306 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 416)  # alias
    buf327 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    buf322 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 416)  # alias
    buf266 = buf265; del buf265  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_51(c_void_p(buf266.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg517_1.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf322.data_ptr()))
    del arg153_1
    del arg154_1
    del arg517_1
    del arg518_1
    del buf258
    del buf259
    del buf260
    del buf261
    del buf262
    del buf263
    del buf264
    # Source Nodes: [bottleneck_output_48, l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf267 = extern_kernels.convolution(buf266, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf267, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg155_1
    del buf266
    buf268 = buf267; del buf267  # reuse
    buf269 = buf256; del buf256  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52(c_void_p(buf268.data_ptr()), c_void_p(arg520_1.data_ptr()), c_void_p(arg521_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf269.data_ptr()))
    del arg156_1
    del arg157_1
    del arg158_1
    del arg520_1
    del arg521_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer7_norm2, l__mod___features_denseblock3_denselayer7_relu2, new_features_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf270 = extern_kernels.convolution(buf268, buf269, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf270, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf268
    buf278 = reinterpret_tensor(buf279, (4, 32, 14, 14), (94080, 1, 6720, 480), 448)  # alias
    buf292 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 448)  # alias
    buf307 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 448)  # alias
    buf323 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 448)  # alias
    buf345 = empty_strided((4, 608, 14, 14), (119168, 1, 8512, 608), device='cpu', dtype=torch.float32)
    buf340 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 448)  # alias
    buf280 = buf279; del buf279  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_53(c_void_p(buf280.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg523_1.data_ptr()), c_void_p(arg524_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf340.data_ptr()))
    del arg159_1
    del arg160_1
    del arg523_1
    del arg524_1
    del buf271
    del buf272
    del buf273
    del buf274
    del buf275
    del buf276
    del buf277
    del buf278
    # Source Nodes: [bottleneck_output_50, l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf281 = extern_kernels.convolution(buf280, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf281, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg161_1
    del buf280
    buf282 = buf281; del buf281  # reuse
    buf283 = buf269; del buf269  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_54(c_void_p(buf282.data_ptr()), c_void_p(arg526_1.data_ptr()), c_void_p(arg527_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf283.data_ptr()))
    del arg162_1
    del arg163_1
    del arg164_1
    del arg526_1
    del arg527_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer8_norm2, l__mod___features_denseblock3_denselayer8_relu2, new_features_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf284 = extern_kernels.convolution(buf282, buf283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf284, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf282
    buf293 = reinterpret_tensor(buf294, (4, 32, 14, 14), (100352, 1, 7168, 512), 480)  # alias
    buf308 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 480)  # alias
    buf324 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 480)  # alias
    buf341 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 480)  # alias
    buf295 = buf294; del buf294  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_55(c_void_p(buf295.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg529_1.data_ptr()), c_void_p(arg530_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg165_1
    del arg166_1
    del arg529_1
    del arg530_1
    del buf285
    del buf286
    del buf287
    del buf288
    del buf289
    del buf290
    del buf291
    del buf292
    del buf293
    # Source Nodes: [bottleneck_output_52, l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf296 = extern_kernels.convolution(buf295, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf296, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg167_1
    del buf295
    buf297 = buf296; del buf296  # reuse
    buf298 = buf283; del buf283  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56(c_void_p(buf297.data_ptr()), c_void_p(arg532_1.data_ptr()), c_void_p(arg533_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg168_1
    del arg169_1
    del arg170_1
    del arg532_1
    del arg533_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer9_norm2, l__mod___features_denseblock3_denselayer9_relu2, new_features_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf299 = extern_kernels.convolution(buf297, buf298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf299, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf297
    buf301 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 256)  # alias
    buf317 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 256)  # alias
    buf334 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 256)  # alias
    buf364 = reinterpret_tensor(buf63, (4, 640, 14, 14), (125440, 1, 8960, 640), 0); del buf63  # reuse
    buf352 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 256)  # alias
    buf302 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 288)  # alias
    buf318 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 288)  # alias
    buf335 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 288)  # alias
    buf353 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 288)  # alias
    buf303 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 320)  # alias
    buf319 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 320)  # alias
    buf336 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 320)  # alias
    buf354 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 320)  # alias
    buf304 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 352)  # alias
    buf320 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 352)  # alias
    buf337 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 352)  # alias
    buf355 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 352)  # alias
    buf309 = reinterpret_tensor(buf310, (4, 32, 14, 14), (106624, 1, 7616, 544), 512)  # alias
    buf325 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 512)  # alias
    buf342 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 512)  # alias
    buf360 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 512)  # alias
    buf311 = buf310; del buf310  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_57(c_void_p(buf311.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(arg535_1.data_ptr()), c_void_p(arg536_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf360.data_ptr()))
    del arg171_1
    del arg172_1
    del arg535_1
    del arg536_1
    del buf300
    del buf301
    del buf302
    del buf303
    del buf304
    del buf305
    del buf306
    del buf307
    del buf308
    del buf309
    # Source Nodes: [bottleneck_output_54, l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf312 = extern_kernels.convolution(buf311, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf312, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg173_1
    del buf311
    buf313 = buf312; del buf312  # reuse
    buf314 = buf298; del buf298  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_58(c_void_p(buf313.data_ptr()), c_void_p(arg538_1.data_ptr()), c_void_p(arg539_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg174_1
    del arg175_1
    del arg176_1
    del arg538_1
    del arg539_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer10_norm2, l__mod___features_denseblock3_denselayer10_relu2, new_features_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf315 = extern_kernels.convolution(buf313, buf314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf315, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf313
    buf316 = reinterpret_tensor(buf327, (4, 256, 14, 14), (112896, 1, 8064, 576), 0)  # alias
    buf333 = reinterpret_tensor(buf345, (4, 256, 14, 14), (119168, 1, 8512, 608), 0)  # alias
    buf351 = reinterpret_tensor(buf364, (4, 256, 14, 14), (125440, 1, 8960, 640), 0)  # alias
    buf384 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf370 = reinterpret_tensor(buf384, (4, 256, 14, 14), (131712, 1, 9408, 672), 0)  # alias
    buf321 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 384)  # alias
    buf338 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 384)  # alias
    buf356 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 384)  # alias
    buf375 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 384)  # alias
    buf326 = reinterpret_tensor(buf327, (4, 32, 14, 14), (112896, 1, 8064, 576), 544)  # alias
    buf343 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 544)  # alias
    buf361 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 544)  # alias
    buf380 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 544)  # alias
    buf328 = buf327; del buf327  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_59(c_void_p(buf328.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(arg541_1.data_ptr()), c_void_p(arg542_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf380.data_ptr()))
    del arg177_1
    del arg178_1
    del arg541_1
    del arg542_1
    del buf316
    del buf317
    del buf318
    del buf319
    del buf320
    del buf321
    del buf322
    del buf323
    del buf324
    del buf325
    del buf326
    # Source Nodes: [bottleneck_output_56, l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf329 = extern_kernels.convolution(buf328, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf329, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg179_1
    del buf328
    buf330 = buf329; del buf329  # reuse
    buf331 = buf314; del buf314  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_60(c_void_p(buf330.data_ptr()), c_void_p(arg544_1.data_ptr()), c_void_p(arg545_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg180_1
    del arg181_1
    del arg182_1
    del arg544_1
    del arg545_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer11_norm2, l__mod___features_denseblock3_denselayer11_relu2, new_features_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf332 = extern_kernels.convolution(buf330, buf331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf332, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf330
    buf339 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 416)  # alias
    buf357 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 416)  # alias
    buf376 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 416)  # alias
    buf405 = empty_strided((4, 704, 14, 14), (137984, 1, 9856, 704), device='cpu', dtype=torch.float32)
    buf396 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 416)  # alias
    buf344 = reinterpret_tensor(buf345, (4, 32, 14, 14), (119168, 1, 8512, 608), 576)  # alias
    buf362 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 576)  # alias
    buf381 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 576)  # alias
    buf401 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 576)  # alias
    buf346 = buf345; del buf345  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_61(c_void_p(buf346.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(arg547_1.data_ptr()), c_void_p(arg548_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg183_1
    del arg184_1
    del arg547_1
    del arg548_1
    del buf333
    del buf334
    del buf335
    del buf336
    del buf337
    del buf338
    del buf339
    del buf340
    del buf341
    del buf342
    del buf343
    del buf344
    # Source Nodes: [bottleneck_output_58, l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf347 = extern_kernels.convolution(buf346, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf347, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg185_1
    del buf346
    buf348 = buf347; del buf347  # reuse
    buf349 = buf331; del buf331  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf348.data_ptr()), c_void_p(arg550_1.data_ptr()), c_void_p(arg551_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(buf349.data_ptr()))
    del arg186_1
    del arg187_1
    del arg188_1
    del arg550_1
    del arg551_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer12_norm2, l__mod___features_denseblock3_denselayer12_relu2, new_features_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf350 = extern_kernels.convolution(buf348, buf349, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf350, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf348
    buf358 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 448)  # alias
    buf377 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 448)  # alias
    buf397 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 448)  # alias
    buf427 = empty_strided((4, 736, 14, 14), (144256, 1, 10304, 736), device='cpu', dtype=torch.float32)
    buf418 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 448)  # alias
    buf359 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 480)  # alias
    buf378 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 480)  # alias
    buf398 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 480)  # alias
    buf419 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 480)  # alias
    buf363 = reinterpret_tensor(buf364, (4, 32, 14, 14), (125440, 1, 8960, 640), 608)  # alias
    buf382 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 608)  # alias
    buf402 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 608)  # alias
    buf423 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 608)  # alias
    buf365 = buf364; del buf364  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_63(c_void_p(buf365.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(arg553_1.data_ptr()), c_void_p(arg554_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf423.data_ptr()))
    del arg189_1
    del arg190_1
    del arg553_1
    del arg554_1
    del buf351
    del buf352
    del buf353
    del buf354
    del buf355
    del buf356
    del buf357
    del buf358
    del buf359
    del buf360
    del buf361
    del buf362
    del buf363
    # Source Nodes: [bottleneck_output_60, l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf366 = extern_kernels.convolution(buf365, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf366, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg191_1
    del buf365
    buf367 = buf366; del buf366  # reuse
    buf368 = buf349; del buf349  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_64(c_void_p(buf367.data_ptr()), c_void_p(arg556_1.data_ptr()), c_void_p(arg557_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(buf368.data_ptr()))
    del arg192_1
    del arg193_1
    del arg194_1
    del arg556_1
    del arg557_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer13_norm2, l__mod___features_denseblock3_denselayer13_relu2, new_features_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf369 = extern_kernels.convolution(buf367, buf368, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf369, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf367
    buf371 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 256)  # alias
    buf391 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 256)  # alias
    buf412 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 256)  # alias
    buf450 = reinterpret_tensor(buf69, (4, 768, 14, 14), (150528, 1, 10752, 768), 0); del buf69  # reuse
    buf434 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 256)  # alias
    buf372 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 288)  # alias
    buf392 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 288)  # alias
    buf413 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 288)  # alias
    buf435 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 288)  # alias
    buf373 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 320)  # alias
    buf393 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 320)  # alias
    buf414 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 320)  # alias
    buf436 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 320)  # alias
    buf374 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 352)  # alias
    buf394 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 352)  # alias
    buf415 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 352)  # alias
    buf437 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 352)  # alias
    buf379 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 512)  # alias
    buf399 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 512)  # alias
    buf420 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 512)  # alias
    buf442 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 512)  # alias
    buf383 = reinterpret_tensor(buf384, (4, 32, 14, 14), (131712, 1, 9408, 672), 640)  # alias
    buf403 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 640)  # alias
    buf424 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 640)  # alias
    buf446 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 640)  # alias
    buf385 = buf384; del buf384  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_65(c_void_p(buf385.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(arg559_1.data_ptr()), c_void_p(arg560_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf446.data_ptr()))
    del arg195_1
    del arg196_1
    del arg559_1
    del arg560_1
    del buf370
    del buf371
    del buf372
    del buf373
    del buf374
    del buf375
    del buf376
    del buf377
    del buf378
    del buf379
    del buf380
    del buf381
    del buf382
    del buf383
    # Source Nodes: [bottleneck_output_62, l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf386 = extern_kernels.convolution(buf385, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf386, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg197_1
    del buf385
    buf387 = buf386; del buf386  # reuse
    buf388 = buf368; del buf368  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_66(c_void_p(buf387.data_ptr()), c_void_p(arg562_1.data_ptr()), c_void_p(arg563_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf388.data_ptr()))
    del arg198_1
    del arg199_1
    del arg200_1
    del arg562_1
    del arg563_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer14_norm2, l__mod___features_denseblock3_denselayer14_relu2, new_features_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf389 = extern_kernels.convolution(buf387, buf388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf389, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf387
    buf390 = reinterpret_tensor(buf405, (4, 256, 14, 14), (137984, 1, 9856, 704), 0)  # alias
    buf411 = reinterpret_tensor(buf427, (4, 256, 14, 14), (144256, 1, 10304, 736), 0)  # alias
    buf433 = reinterpret_tensor(buf450, (4, 256, 14, 14), (150528, 1, 10752, 768), 0)  # alias
    buf395 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 384)  # alias
    buf416 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 384)  # alias
    buf438 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 384)  # alias
    buf400 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 544)  # alias
    buf421 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 544)  # alias
    buf443 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 544)  # alias
    buf404 = reinterpret_tensor(buf405, (4, 32, 14, 14), (137984, 1, 9856, 704), 672)  # alias
    buf425 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 672)  # alias
    buf447 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 672)  # alias
    buf406 = buf405; del buf405  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_67(c_void_p(buf406.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(arg565_1.data_ptr()), c_void_p(arg566_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf447.data_ptr()))
    del arg201_1
    del arg202_1
    del arg565_1
    del arg566_1
    del buf390
    del buf391
    del buf392
    del buf393
    del buf394
    del buf395
    del buf396
    del buf397
    del buf398
    del buf399
    del buf400
    del buf401
    del buf402
    del buf403
    del buf404
    # Source Nodes: [bottleneck_output_64, l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf407 = extern_kernels.convolution(buf406, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf407, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg203_1
    del buf406
    buf408 = buf407; del buf407  # reuse
    buf409 = buf388; del buf388  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68(c_void_p(buf408.data_ptr()), c_void_p(arg568_1.data_ptr()), c_void_p(arg569_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(buf409.data_ptr()))
    del arg204_1
    del arg205_1
    del arg206_1
    del arg568_1
    del arg569_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer15_norm2, l__mod___features_denseblock3_denselayer15_relu2, new_features_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf410 = extern_kernels.convolution(buf408, buf409, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf410, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf408
    buf417 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 416)  # alias
    buf439 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 416)  # alias
    buf474 = empty_strided((4, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    buf462 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 416)  # alias
    buf422 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 576)  # alias
    buf444 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 576)  # alias
    buf467 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 576)  # alias
    buf426 = reinterpret_tensor(buf427, (4, 32, 14, 14), (144256, 1, 10304, 736), 704)  # alias
    buf448 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 704)  # alias
    buf471 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 704)  # alias
    buf428 = buf427; del buf427  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_69(c_void_p(buf428.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(arg571_1.data_ptr()), c_void_p(arg572_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf471.data_ptr()))
    del arg207_1
    del arg208_1
    del arg571_1
    del arg572_1
    del buf411
    del buf412
    del buf413
    del buf414
    del buf415
    del buf416
    del buf417
    del buf418
    del buf419
    del buf420
    del buf421
    del buf422
    del buf423
    del buf424
    del buf425
    del buf426
    # Source Nodes: [bottleneck_output_66, l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf429 = extern_kernels.convolution(buf428, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf429, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg209_1
    del buf428
    buf430 = buf429; del buf429  # reuse
    buf431 = buf409; del buf409  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_70(c_void_p(buf430.data_ptr()), c_void_p(arg574_1.data_ptr()), c_void_p(arg575_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf431.data_ptr()))
    del arg210_1
    del arg211_1
    del arg212_1
    del arg574_1
    del arg575_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer16_norm2, l__mod___features_denseblock3_denselayer16_relu2, new_features_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf432 = extern_kernels.convolution(buf430, buf431, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf432, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf430
    buf440 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 448)  # alias
    buf463 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 448)  # alias
    buf499 = empty_strided((4, 832, 14, 14), (163072, 1, 11648, 832), device='cpu', dtype=torch.float32)
    buf487 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 448)  # alias
    buf441 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 480)  # alias
    buf464 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 480)  # alias
    buf488 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 480)  # alias
    buf445 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 608)  # alias
    buf468 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 608)  # alias
    buf492 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 608)  # alias
    buf449 = reinterpret_tensor(buf450, (4, 32, 14, 14), (150528, 1, 10752, 768), 736)  # alias
    buf472 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 736)  # alias
    buf496 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 736)  # alias
    buf451 = buf450; del buf450  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_71(c_void_p(buf451.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(arg577_1.data_ptr()), c_void_p(arg578_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf496.data_ptr()))
    del arg213_1
    del arg214_1
    del arg577_1
    del arg578_1
    del buf433
    del buf434
    del buf435
    del buf436
    del buf437
    del buf438
    del buf439
    del buf440
    del buf441
    del buf442
    del buf443
    del buf444
    del buf445
    del buf446
    del buf447
    del buf448
    del buf449
    # Source Nodes: [bottleneck_output_68, l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf452 = extern_kernels.convolution(buf451, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf452, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg215_1
    del buf451
    buf453 = buf452; del buf452  # reuse
    buf454 = buf431; del buf431  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72(c_void_p(buf453.data_ptr()), c_void_p(arg580_1.data_ptr()), c_void_p(arg581_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf454.data_ptr()))
    del arg216_1
    del arg217_1
    del arg218_1
    del arg580_1
    del arg581_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer17_norm2, l__mod___features_denseblock3_denselayer17_relu2, new_features_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf455 = extern_kernels.convolution(buf453, buf454, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf455, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf453
    buf456 = reinterpret_tensor(buf474, (4, 256, 14, 14), (156800, 1, 11200, 800), 0)  # alias
    buf480 = reinterpret_tensor(buf499, (4, 256, 14, 14), (163072, 1, 11648, 832), 0)  # alias
    buf525 = empty_strided((4, 864, 14, 14), (169344, 1, 12096, 864), device='cpu', dtype=torch.float32)
    buf505 = reinterpret_tensor(buf525, (4, 256, 14, 14), (169344, 1, 12096, 864), 0)  # alias
    buf457 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 256)  # alias
    buf481 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 256)  # alias
    buf506 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 256)  # alias
    buf458 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 288)  # alias
    buf482 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 288)  # alias
    buf507 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 288)  # alias
    buf459 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 320)  # alias
    buf483 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 320)  # alias
    buf508 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 320)  # alias
    buf460 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 352)  # alias
    buf484 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 352)  # alias
    buf509 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 352)  # alias
    buf461 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 384)  # alias
    buf485 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 384)  # alias
    buf510 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 384)  # alias
    buf465 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 512)  # alias
    buf489 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 512)  # alias
    buf514 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 512)  # alias
    buf466 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 544)  # alias
    buf490 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 544)  # alias
    buf515 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 544)  # alias
    buf469 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 640)  # alias
    buf493 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 640)  # alias
    buf518 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 640)  # alias
    buf470 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 672)  # alias
    buf494 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 672)  # alias
    buf519 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 672)  # alias
    buf473 = reinterpret_tensor(buf474, (4, 32, 14, 14), (156800, 1, 11200, 800), 768)  # alias
    buf497 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 768)  # alias
    buf522 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 768)  # alias
    buf475 = buf474; del buf474  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_73(c_void_p(buf475.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(arg583_1.data_ptr()), c_void_p(arg584_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf522.data_ptr()))
    del arg219_1
    del arg220_1
    del arg583_1
    del arg584_1
    del buf456
    del buf457
    del buf458
    del buf459
    del buf460
    del buf461
    del buf462
    del buf463
    del buf464
    del buf465
    del buf466
    del buf467
    del buf468
    del buf469
    del buf470
    del buf471
    del buf472
    del buf473
    # Source Nodes: [bottleneck_output_70, l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf476 = extern_kernels.convolution(buf475, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf476, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg221_1
    del buf475
    buf477 = buf476; del buf476  # reuse
    buf478 = buf454; del buf454  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_74(c_void_p(buf477.data_ptr()), c_void_p(arg586_1.data_ptr()), c_void_p(arg587_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf478.data_ptr()))
    del arg222_1
    del arg223_1
    del arg224_1
    del arg586_1
    del arg587_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer18_norm2, l__mod___features_denseblock3_denselayer18_relu2, new_features_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf479 = extern_kernels.convolution(buf477, buf478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf479, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf477
    buf486 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 416)  # alias
    buf511 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 416)  # alias
    buf552 = reinterpret_tensor(buf75, (4, 896, 14, 14), (175616, 1, 12544, 896), 0); del buf75  # reuse
    buf537 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 416)  # alias
    buf491 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 576)  # alias
    buf516 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 576)  # alias
    buf542 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 576)  # alias
    buf495 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 704)  # alias
    buf520 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 704)  # alias
    buf546 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 704)  # alias
    buf498 = reinterpret_tensor(buf499, (4, 32, 14, 14), (163072, 1, 11648, 832), 800)  # alias
    buf523 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 800)  # alias
    buf549 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 800)  # alias
    buf500 = buf499; del buf499  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_75(c_void_p(buf500.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(arg589_1.data_ptr()), c_void_p(arg590_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf549.data_ptr()))
    del arg225_1
    del arg226_1
    del arg589_1
    del arg590_1
    del buf480
    del buf481
    del buf482
    del buf483
    del buf484
    del buf485
    del buf486
    del buf487
    del buf488
    del buf489
    del buf490
    del buf491
    del buf492
    del buf493
    del buf494
    del buf495
    del buf496
    del buf497
    del buf498
    # Source Nodes: [bottleneck_output_72, l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf501 = extern_kernels.convolution(buf500, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf501, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg227_1
    del buf500
    buf502 = buf501; del buf501  # reuse
    buf503 = buf478; del buf478  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_76(c_void_p(buf502.data_ptr()), c_void_p(arg592_1.data_ptr()), c_void_p(arg593_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf503.data_ptr()))
    del arg228_1
    del arg229_1
    del arg230_1
    del arg592_1
    del arg593_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer19_norm2, l__mod___features_denseblock3_denselayer19_relu2, new_features_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf504 = extern_kernels.convolution(buf502, buf503, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf504, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf502
    buf512 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 448)  # alias
    buf538 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 448)  # alias
    buf580 = empty_strided((4, 928, 14, 14), (181888, 1, 12992, 928), device='cpu', dtype=torch.float32)
    buf565 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 448)  # alias
    buf513 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 480)  # alias
    buf539 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 480)  # alias
    buf566 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 480)  # alias
    buf517 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 608)  # alias
    buf543 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 608)  # alias
    buf570 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 608)  # alias
    buf521 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 736)  # alias
    buf547 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 736)  # alias
    buf574 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 736)  # alias
    buf524 = reinterpret_tensor(buf525, (4, 32, 14, 14), (169344, 1, 12096, 864), 832)  # alias
    buf550 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 832)  # alias
    buf577 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 832)  # alias
    buf526 = buf525; del buf525  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_77(c_void_p(buf526.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(arg595_1.data_ptr()), c_void_p(arg596_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf577.data_ptr()))
    del arg231_1
    del arg232_1
    del arg595_1
    del arg596_1
    del buf505
    del buf506
    del buf507
    del buf508
    del buf509
    del buf510
    del buf511
    del buf512
    del buf513
    del buf514
    del buf515
    del buf516
    del buf517
    del buf518
    del buf519
    del buf520
    del buf521
    del buf522
    del buf523
    del buf524
    # Source Nodes: [bottleneck_output_74, l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf527 = extern_kernels.convolution(buf526, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf527, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg233_1
    del buf526
    buf528 = buf527; del buf527  # reuse
    buf529 = buf503; del buf503  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_78(c_void_p(buf528.data_ptr()), c_void_p(arg598_1.data_ptr()), c_void_p(arg599_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf529.data_ptr()))
    del arg234_1
    del arg235_1
    del arg236_1
    del arg598_1
    del arg599_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer20_norm2, l__mod___features_denseblock3_denselayer20_relu2, new_features_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf530 = extern_kernels.convolution(buf528, buf529, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf530, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf528
    buf531 = reinterpret_tensor(buf552, (4, 256, 14, 14), (175616, 1, 12544, 896), 0)  # alias
    buf558 = reinterpret_tensor(buf580, (4, 256, 14, 14), (181888, 1, 12992, 928), 0)  # alias
    buf609 = empty_strided((4, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    buf586 = reinterpret_tensor(buf609, (4, 256, 14, 14), (188160, 1, 13440, 960), 0)  # alias
    buf532 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 256)  # alias
    buf559 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 256)  # alias
    buf587 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 256)  # alias
    buf533 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 288)  # alias
    buf560 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 288)  # alias
    buf588 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 288)  # alias
    buf534 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 320)  # alias
    buf561 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 320)  # alias
    buf589 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 320)  # alias
    buf535 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 352)  # alias
    buf562 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 352)  # alias
    buf590 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 352)  # alias
    buf536 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 384)  # alias
    buf563 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 384)  # alias
    buf591 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 384)  # alias
    buf540 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 512)  # alias
    buf567 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 512)  # alias
    buf595 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 512)  # alias
    buf541 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 544)  # alias
    buf568 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 544)  # alias
    buf596 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 544)  # alias
    buf544 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 640)  # alias
    buf571 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 640)  # alias
    buf599 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 640)  # alias
    buf545 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 672)  # alias
    buf572 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 672)  # alias
    buf600 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 672)  # alias
    buf548 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 768)  # alias
    buf575 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 768)  # alias
    buf603 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 768)  # alias
    buf551 = reinterpret_tensor(buf552, (4, 32, 14, 14), (175616, 1, 12544, 896), 864)  # alias
    buf578 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 864)  # alias
    buf606 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 864)  # alias
    buf553 = buf552; del buf552  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_79(c_void_p(buf553.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(arg601_1.data_ptr()), c_void_p(arg602_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf606.data_ptr()))
    del arg237_1
    del arg238_1
    del arg601_1
    del arg602_1
    del buf531
    del buf532
    del buf533
    del buf534
    del buf535
    del buf536
    del buf537
    del buf538
    del buf539
    del buf540
    del buf541
    del buf542
    del buf543
    del buf544
    del buf545
    del buf546
    del buf547
    del buf548
    del buf549
    del buf550
    del buf551
    # Source Nodes: [bottleneck_output_76, l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf554 = extern_kernels.convolution(buf553, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf554, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg239_1
    del buf553
    buf555 = buf554; del buf554  # reuse
    buf556 = buf529; del buf529  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_80(c_void_p(buf555.data_ptr()), c_void_p(arg604_1.data_ptr()), c_void_p(arg605_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf556.data_ptr()))
    del arg240_1
    del arg241_1
    del arg242_1
    del arg604_1
    del arg605_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer21_norm2, l__mod___features_denseblock3_denselayer21_relu2, new_features_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf557 = extern_kernels.convolution(buf555, buf556, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf557, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf555
    buf564 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 416)  # alias
    buf592 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 416)  # alias
    buf639 = empty_strided((4, 992, 14, 14), (194432, 1, 13888, 992), device='cpu', dtype=torch.float32)
    buf621 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 416)  # alias
    buf569 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 576)  # alias
    buf597 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 576)  # alias
    buf626 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 576)  # alias
    buf573 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 704)  # alias
    buf601 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 704)  # alias
    buf630 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 704)  # alias
    buf576 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 800)  # alias
    buf604 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 800)  # alias
    buf633 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 800)  # alias
    buf579 = reinterpret_tensor(buf580, (4, 32, 14, 14), (181888, 1, 12992, 928), 896)  # alias
    buf607 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 896)  # alias
    buf636 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 896)  # alias
    buf581 = buf580; del buf580  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_81(c_void_p(buf581.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(arg607_1.data_ptr()), c_void_p(arg608_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf636.data_ptr()))
    del arg243_1
    del arg244_1
    del arg607_1
    del arg608_1
    del buf558
    del buf559
    del buf560
    del buf561
    del buf562
    del buf563
    del buf564
    del buf565
    del buf566
    del buf567
    del buf568
    del buf569
    del buf570
    del buf571
    del buf572
    del buf573
    del buf574
    del buf575
    del buf576
    del buf577
    del buf578
    del buf579
    # Source Nodes: [bottleneck_output_78, l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf582 = extern_kernels.convolution(buf581, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf582, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg245_1
    del buf581
    buf583 = buf582; del buf582  # reuse
    buf584 = buf556; del buf556  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_82(c_void_p(buf583.data_ptr()), c_void_p(arg610_1.data_ptr()), c_void_p(arg611_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf584.data_ptr()))
    del arg246_1
    del arg247_1
    del arg248_1
    del arg610_1
    del arg611_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer22_norm2, l__mod___features_denseblock3_denselayer22_relu2, new_features_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf585 = extern_kernels.convolution(buf583, buf584, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf585, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf583
    buf593 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 448)  # alias
    buf622 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 448)  # alias
    buf670 = reinterpret_tensor(buf211, (4, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf211  # reuse
    buf652 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 448)  # alias
    buf594 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 480)  # alias
    buf623 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 480)  # alias
    buf653 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 480)  # alias
    buf598 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 608)  # alias
    buf627 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 608)  # alias
    buf657 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 608)  # alias
    buf602 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 736)  # alias
    buf631 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 736)  # alias
    buf661 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 736)  # alias
    buf605 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 832)  # alias
    buf634 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 832)  # alias
    buf664 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 832)  # alias
    buf608 = reinterpret_tensor(buf609, (4, 32, 14, 14), (188160, 1, 13440, 960), 928)  # alias
    buf637 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 928)  # alias
    buf667 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 928)  # alias
    buf610 = buf609; del buf609  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_83(c_void_p(buf610.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(arg613_1.data_ptr()), c_void_p(arg614_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf667.data_ptr()))
    del arg249_1
    del arg250_1
    del arg613_1
    del arg614_1
    del buf270
    del buf284
    del buf350
    del buf432
    del buf504
    del buf585
    del buf586
    del buf587
    del buf588
    del buf589
    del buf590
    del buf591
    del buf592
    del buf593
    del buf594
    del buf595
    del buf596
    del buf597
    del buf598
    del buf599
    del buf600
    del buf601
    del buf602
    del buf603
    del buf604
    del buf605
    del buf606
    del buf607
    del buf608
    # Source Nodes: [bottleneck_output_80, l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf611 = extern_kernels.convolution(buf610, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf611, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg251_1
    del buf610
    buf612 = buf611; del buf611  # reuse
    buf613 = buf584; del buf584  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_84(c_void_p(buf612.data_ptr()), c_void_p(arg616_1.data_ptr()), c_void_p(arg617_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(buf613.data_ptr()))
    del arg252_1
    del arg253_1
    del arg254_1
    del arg616_1
    del arg617_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer23_norm2, l__mod___features_denseblock3_denselayer23_relu2, new_features_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf614 = extern_kernels.convolution(buf612, buf613, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf614, (4, 32, 14, 14), (6272, 1, 448, 32))
    del buf612
    buf615 = reinterpret_tensor(buf639, (4, 256, 14, 14), (194432, 1, 13888, 992), 0)  # alias
    buf645 = reinterpret_tensor(buf670, (4, 256, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
    buf616 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 256)  # alias
    buf646 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 256)  # alias
    buf617 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 288)  # alias
    buf647 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 288)  # alias
    buf618 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 320)  # alias
    buf648 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 320)  # alias
    buf619 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 352)  # alias
    buf649 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 352)  # alias
    buf620 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 384)  # alias
    buf650 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 384)  # alias
    buf624 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 512)  # alias
    buf654 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
    buf625 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 544)  # alias
    buf655 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 544)  # alias
    buf628 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 640)  # alias
    buf658 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 640)  # alias
    buf629 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 672)  # alias
    buf659 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 672)  # alias
    buf632 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 768)  # alias
    buf662 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 768)  # alias
    buf635 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 864)  # alias
    buf665 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 864)  # alias
    buf638 = reinterpret_tensor(buf639, (4, 32, 14, 14), (194432, 1, 13888, 992), 960)  # alias
    buf668 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 960)  # alias
    buf640 = buf639; del buf639  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_85(c_void_p(buf640.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(arg619_1.data_ptr()), c_void_p(arg620_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf668.data_ptr()))
    del arg255_1
    del arg256_1
    del arg619_1
    del arg620_1
    del buf216
    del buf222
    del buf228
    del buf234
    del buf235
    del buf240
    del buf245
    del buf299
    del buf315
    del buf369
    del buf389
    del buf455
    del buf530
    del buf614
    del buf615
    del buf616
    del buf617
    del buf618
    del buf619
    del buf620
    del buf621
    del buf622
    del buf623
    del buf624
    del buf625
    del buf626
    del buf627
    del buf628
    del buf629
    del buf630
    del buf631
    del buf632
    del buf633
    del buf634
    del buf635
    del buf636
    del buf637
    del buf638
    # Source Nodes: [bottleneck_output_82, l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf641 = extern_kernels.convolution(buf640, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf641, (4, 128, 14, 14), (25088, 1, 1792, 128))
    del arg257_1
    del buf640
    buf642 = buf641; del buf641  # reuse
    buf643 = buf613; del buf613  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86(c_void_p(buf642.data_ptr()), c_void_p(arg622_1.data_ptr()), c_void_p(arg623_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(buf643.data_ptr()))
    del arg258_1
    del arg259_1
    del arg260_1
    del arg622_1
    del arg623_1
    # Source Nodes: [l__mod___features_denseblock3_denselayer24_norm2, l__mod___features_denseblock3_denselayer24_relu2, new_features_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf644 = extern_kernels.convolution(buf642, buf643, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf644, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf651 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 416)  # alias
    buf656 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 576)  # alias
    buf660 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 704)  # alias
    buf663 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 800)  # alias
    buf666 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 896)  # alias
    buf669 = reinterpret_tensor(buf670, (4, 32, 14, 14), (200704, 1, 14336, 1024), 992)  # alias
    buf671 = buf670; del buf670  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_87(c_void_p(buf671.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(arg625_1.data_ptr()), c_void_p(arg626_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf669.data_ptr()))
    del arg261_1
    del arg262_1
    del arg625_1
    del arg626_1
    del buf257
    del buf332
    del buf410
    del buf479
    del buf557
    del buf644
    del buf645
    del buf646
    del buf647
    del buf648
    del buf649
    del buf650
    del buf651
    del buf652
    del buf653
    del buf654
    del buf655
    del buf656
    del buf657
    del buf658
    del buf659
    del buf660
    del buf661
    del buf662
    del buf663
    del buf664
    del buf665
    del buf666
    del buf667
    del buf668
    del buf669
    # Source Nodes: [l__mod___features_transition3_conv, l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf672 = extern_kernels.convolution(buf671, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf672, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del arg263_1
    del buf671
    buf673 = reinterpret_tensor(buf642, (4, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf642  # reuse
    buf701 = empty_strided((4, 640, 7, 7), (31360, 1, 4480, 640), device='cpu', dtype=torch.float32)
    buf696 = reinterpret_tensor(buf701, (4, 512, 7, 7), (31360, 1, 4480, 640), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_88(c_void_p(buf672.data_ptr()), c_void_p(arg628_1.data_ptr()), c_void_p(arg629_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf696.data_ptr()))
    del arg264_1
    del arg265_1
    del arg628_1
    del arg629_1
    # Source Nodes: [bottleneck_output_84, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
    buf674 = extern_kernels.convolution(buf673, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf674, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg266_1
    del buf673
    buf675 = buf674; del buf674  # reuse
    buf676 = buf643; del buf643  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_89(c_void_p(buf675.data_ptr()), c_void_p(arg631_1.data_ptr()), c_void_p(arg632_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf676.data_ptr()))
    del arg267_1
    del arg268_1
    del arg269_1
    del arg631_1
    del arg632_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2, new_features_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf677 = extern_kernels.convolution(buf675, buf676, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf677, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf675
    buf678 = empty_strided((4, 544, 7, 7), (26656, 1, 3808, 544), device='cpu', dtype=torch.float32)
    buf679 = buf678; del buf678  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_90(c_void_p(buf679.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(arg634_1.data_ptr()), c_void_p(arg635_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()))
    del arg270_1
    del arg271_1
    del arg634_1
    del arg635_1
    # Source Nodes: [bottleneck_output_86, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
    buf680 = extern_kernels.convolution(buf679, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf680, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg272_1
    del buf679
    buf681 = buf680; del buf680  # reuse
    buf682 = buf676; del buf676  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_91(c_void_p(buf681.data_ptr()), c_void_p(arg637_1.data_ptr()), c_void_p(arg638_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf682.data_ptr()))
    del arg273_1
    del arg274_1
    del arg275_1
    del arg637_1
    del arg638_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer2_norm2, l__mod___features_denseblock4_denselayer2_relu2, new_features_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf683 = extern_kernels.convolution(buf681, buf682, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf683, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf681
    buf684 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    buf685 = buf684; del buf684  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_92(c_void_p(buf685.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(arg640_1.data_ptr()), c_void_p(arg641_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()))
    del arg276_1
    del arg277_1
    del arg640_1
    del arg641_1
    # Source Nodes: [bottleneck_output_88, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf686 = extern_kernels.convolution(buf685, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf686, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg278_1
    del buf685
    buf687 = buf686; del buf686  # reuse
    buf688 = buf682; del buf682  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_93(c_void_p(buf687.data_ptr()), c_void_p(arg643_1.data_ptr()), c_void_p(arg644_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf688.data_ptr()))
    del arg279_1
    del arg280_1
    del arg281_1
    del arg643_1
    del arg644_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer3_norm2, l__mod___features_denseblock4_denselayer3_relu2, new_features_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf689 = extern_kernels.convolution(buf687, buf688, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf689, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf687
    buf690 = empty_strided((4, 608, 7, 7), (29792, 1, 4256, 608), device='cpu', dtype=torch.float32)
    buf691 = buf690; del buf690  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_94(c_void_p(buf691.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(arg646_1.data_ptr()), c_void_p(arg647_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()))
    del arg282_1
    del arg283_1
    del arg646_1
    del arg647_1
    del buf672
    # Source Nodes: [bottleneck_output_90, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf692 = extern_kernels.convolution(buf691, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf692, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg284_1
    del buf691
    buf693 = buf692; del buf692  # reuse
    buf694 = buf688; del buf688  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_95(c_void_p(buf693.data_ptr()), c_void_p(arg649_1.data_ptr()), c_void_p(arg650_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf694.data_ptr()))
    del arg285_1
    del arg286_1
    del arg287_1
    del arg649_1
    del arg650_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer4_norm2, l__mod___features_denseblock4_denselayer4_relu2, new_features_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf695 = extern_kernels.convolution(buf693, buf694, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf695, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf693
    buf697 = reinterpret_tensor(buf701, (4, 32, 7, 7), (31360, 1, 4480, 640), 512)  # alias
    buf713 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    buf708 = reinterpret_tensor(buf713, (4, 32, 7, 7), (32928, 1, 4704, 672), 512)  # alias
    buf726 = empty_strided((4, 704, 7, 7), (34496, 1, 4928, 704), device='cpu', dtype=torch.float32)
    buf720 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 512)  # alias
    buf740 = empty_strided((4, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    buf733 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 512)  # alias
    buf755 = empty_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf747 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 512)  # alias
    buf698 = reinterpret_tensor(buf701, (4, 32, 7, 7), (31360, 1, 4480, 640), 544)  # alias
    buf709 = reinterpret_tensor(buf713, (4, 32, 7, 7), (32928, 1, 4704, 672), 544)  # alias
    buf721 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 544)  # alias
    buf734 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 544)  # alias
    buf748 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 544)  # alias
    buf699 = reinterpret_tensor(buf701, (4, 32, 7, 7), (31360, 1, 4480, 640), 576)  # alias
    buf710 = reinterpret_tensor(buf713, (4, 32, 7, 7), (32928, 1, 4704, 672), 576)  # alias
    buf722 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 576)  # alias
    buf735 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 576)  # alias
    buf749 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 576)  # alias
    buf700 = reinterpret_tensor(buf701, (4, 32, 7, 7), (31360, 1, 4480, 640), 608)  # alias
    buf711 = reinterpret_tensor(buf713, (4, 32, 7, 7), (32928, 1, 4704, 672), 608)  # alias
    buf723 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 608)  # alias
    buf736 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 608)  # alias
    buf750 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 608)  # alias
    buf702 = empty_strided((4, 640, 7, 7), (31360, 1, 4480, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_96(c_void_p(buf677.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(arg652_1.data_ptr()), c_void_p(arg653_1.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf702.data_ptr()))
    del arg288_1
    del arg289_1
    del arg652_1
    del arg653_1
    del buf697
    del buf698
    del buf699
    del buf700
    # Source Nodes: [bottleneck_output_92, l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf703 = extern_kernels.convolution(buf702, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf703, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg290_1
    del buf702
    buf704 = buf703; del buf703  # reuse
    buf705 = buf694; del buf694  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_97(c_void_p(buf704.data_ptr()), c_void_p(arg655_1.data_ptr()), c_void_p(arg656_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf705.data_ptr()))
    del arg291_1
    del arg292_1
    del arg293_1
    del arg655_1
    del arg656_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer5_norm2, l__mod___features_denseblock4_denselayer5_relu2, new_features_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf706 = extern_kernels.convolution(buf704, buf705, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf706, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf704
    buf707 = reinterpret_tensor(buf713, (4, 512, 7, 7), (32928, 1, 4704, 672), 0)  # alias
    buf719 = reinterpret_tensor(buf726, (4, 512, 7, 7), (34496, 1, 4928, 704), 0)  # alias
    buf732 = reinterpret_tensor(buf740, (4, 512, 7, 7), (36064, 1, 5152, 736), 0)  # alias
    buf746 = reinterpret_tensor(buf755, (4, 512, 7, 7), (37632, 1, 5376, 768), 0)  # alias
    buf771 = empty_strided((4, 800, 7, 7), (39200, 1, 5600, 800), device='cpu', dtype=torch.float32)
    buf761 = reinterpret_tensor(buf771, (4, 512, 7, 7), (39200, 1, 5600, 800), 0)  # alias
    buf712 = reinterpret_tensor(buf713, (4, 32, 7, 7), (32928, 1, 4704, 672), 640)  # alias
    buf724 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 640)  # alias
    buf737 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 640)  # alias
    buf751 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 640)  # alias
    buf766 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 640)  # alias
    buf714 = buf713; del buf713  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_98(c_void_p(buf714.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(arg658_1.data_ptr()), c_void_p(arg659_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf766.data_ptr()))
    del arg294_1
    del arg295_1
    del arg658_1
    del arg659_1
    del buf707
    del buf708
    del buf709
    del buf710
    del buf711
    del buf712
    # Source Nodes: [bottleneck_output_94, l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf715 = extern_kernels.convolution(buf714, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf715, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg296_1
    del buf714
    buf716 = buf715; del buf715  # reuse
    buf717 = buf705; del buf705  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_99(c_void_p(buf716.data_ptr()), c_void_p(arg661_1.data_ptr()), c_void_p(arg662_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf717.data_ptr()))
    del arg297_1
    del arg298_1
    del arg299_1
    del arg661_1
    del arg662_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer6_norm2, l__mod___features_denseblock4_denselayer6_relu2, new_features_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf718 = extern_kernels.convolution(buf716, buf717, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf718, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf716
    buf725 = reinterpret_tensor(buf726, (4, 32, 7, 7), (34496, 1, 4928, 704), 672)  # alias
    buf738 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 672)  # alias
    buf752 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 672)  # alias
    buf767 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 672)  # alias
    buf788 = empty_strided((4, 832, 7, 7), (40768, 1, 5824, 832), device='cpu', dtype=torch.float32)
    buf783 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 672)  # alias
    buf727 = buf726; del buf726  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_100(c_void_p(buf727.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(arg664_1.data_ptr()), c_void_p(arg665_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf783.data_ptr()))
    del arg300_1
    del arg301_1
    del arg664_1
    del arg665_1
    del buf719
    del buf720
    del buf721
    del buf722
    del buf723
    del buf724
    del buf725
    # Source Nodes: [bottleneck_output_96, l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf728 = extern_kernels.convolution(buf727, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf728, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg302_1
    del buf727
    buf729 = buf728; del buf728  # reuse
    buf730 = buf717; del buf717  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_101(c_void_p(buf729.data_ptr()), c_void_p(arg667_1.data_ptr()), c_void_p(arg668_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(buf730.data_ptr()))
    del arg303_1
    del arg304_1
    del arg305_1
    del arg667_1
    del arg668_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer7_norm2, l__mod___features_denseblock4_denselayer7_relu2, new_features_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf731 = extern_kernels.convolution(buf729, buf730, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf731, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf729
    buf739 = reinterpret_tensor(buf740, (4, 32, 7, 7), (36064, 1, 5152, 736), 704)  # alias
    buf753 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 704)  # alias
    buf768 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 704)  # alias
    buf784 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 704)  # alias
    buf806 = empty_strided((4, 864, 7, 7), (42336, 1, 6048, 864), device='cpu', dtype=torch.float32)
    buf801 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 704)  # alias
    buf741 = buf740; del buf740  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_102(c_void_p(buf741.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(arg670_1.data_ptr()), c_void_p(arg671_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf801.data_ptr()))
    del arg306_1
    del arg307_1
    del arg670_1
    del arg671_1
    del buf732
    del buf733
    del buf734
    del buf735
    del buf736
    del buf737
    del buf738
    del buf739
    # Source Nodes: [bottleneck_output_98, l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf742 = extern_kernels.convolution(buf741, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf742, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg308_1
    del buf741
    buf743 = buf742; del buf742  # reuse
    buf744 = buf730; del buf730  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_103(c_void_p(buf743.data_ptr()), c_void_p(arg673_1.data_ptr()), c_void_p(arg674_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(buf744.data_ptr()))
    del arg309_1
    del arg310_1
    del arg311_1
    del arg673_1
    del arg674_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer8_norm2, l__mod___features_denseblock4_denselayer8_relu2, new_features_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf745 = extern_kernels.convolution(buf743, buf744, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf745, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf743
    buf754 = reinterpret_tensor(buf755, (4, 32, 7, 7), (37632, 1, 5376, 768), 736)  # alias
    buf769 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 736)  # alias
    buf785 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 736)  # alias
    buf802 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 736)  # alias
    buf756 = buf755; del buf755  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_104(c_void_p(buf756.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(arg676_1.data_ptr()), c_void_p(arg677_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf802.data_ptr()))
    del arg312_1
    del arg313_1
    del arg676_1
    del arg677_1
    del buf746
    del buf747
    del buf748
    del buf749
    del buf750
    del buf751
    del buf752
    del buf753
    del buf754
    # Source Nodes: [bottleneck_output_100, l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf757 = extern_kernels.convolution(buf756, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf757, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg314_1
    del buf756
    buf758 = buf757; del buf757  # reuse
    buf759 = buf744; del buf744  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_105(c_void_p(buf758.data_ptr()), c_void_p(arg679_1.data_ptr()), c_void_p(arg680_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf759.data_ptr()))
    del arg315_1
    del arg316_1
    del arg317_1
    del arg679_1
    del arg680_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer9_norm2, l__mod___features_denseblock4_denselayer9_relu2, new_features_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf760 = extern_kernels.convolution(buf758, buf759, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf760, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf758
    buf762 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 512)  # alias
    buf778 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 512)  # alias
    buf795 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 512)  # alias
    buf825 = empty_strided((4, 896, 7, 7), (43904, 1, 6272, 896), device='cpu', dtype=torch.float32)
    buf813 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 512)  # alias
    buf763 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 544)  # alias
    buf779 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 544)  # alias
    buf796 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 544)  # alias
    buf814 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 544)  # alias
    buf764 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 576)  # alias
    buf780 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 576)  # alias
    buf797 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 576)  # alias
    buf815 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 576)  # alias
    buf765 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 608)  # alias
    buf781 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 608)  # alias
    buf798 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 608)  # alias
    buf816 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 608)  # alias
    buf770 = reinterpret_tensor(buf771, (4, 32, 7, 7), (39200, 1, 5600, 800), 768)  # alias
    buf786 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 768)  # alias
    buf803 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 768)  # alias
    buf821 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 768)  # alias
    buf772 = buf771; del buf771  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_106(c_void_p(buf772.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(arg682_1.data_ptr()), c_void_p(arg683_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(buf821.data_ptr()))
    del arg318_1
    del arg319_1
    del arg682_1
    del arg683_1
    del buf761
    del buf762
    del buf763
    del buf764
    del buf765
    del buf766
    del buf767
    del buf768
    del buf769
    del buf770
    # Source Nodes: [bottleneck_output_102, l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf773 = extern_kernels.convolution(buf772, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf773, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg320_1
    del buf772
    buf774 = buf773; del buf773  # reuse
    buf775 = buf759; del buf759  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_107(c_void_p(buf774.data_ptr()), c_void_p(arg685_1.data_ptr()), c_void_p(arg686_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(buf775.data_ptr()))
    del arg321_1
    del arg322_1
    del arg323_1
    del arg685_1
    del arg686_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer10_norm2, l__mod___features_denseblock4_denselayer10_relu2, new_features_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf776 = extern_kernels.convolution(buf774, buf775, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf776, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf774
    buf777 = reinterpret_tensor(buf788, (4, 512, 7, 7), (40768, 1, 5824, 832), 0)  # alias
    buf794 = reinterpret_tensor(buf806, (4, 512, 7, 7), (42336, 1, 6048, 864), 0)  # alias
    buf812 = reinterpret_tensor(buf825, (4, 512, 7, 7), (43904, 1, 6272, 896), 0)  # alias
    buf845 = empty_strided((4, 928, 7, 7), (45472, 1, 6496, 928), device='cpu', dtype=torch.float32)
    buf831 = reinterpret_tensor(buf845, (4, 512, 7, 7), (45472, 1, 6496, 928), 0)  # alias
    buf782 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 640)  # alias
    buf799 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 640)  # alias
    buf817 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 640)  # alias
    buf836 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 640)  # alias
    buf787 = reinterpret_tensor(buf788, (4, 32, 7, 7), (40768, 1, 5824, 832), 800)  # alias
    buf804 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 800)  # alias
    buf822 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 800)  # alias
    buf841 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 800)  # alias
    buf789 = buf788; del buf788  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_108(c_void_p(buf789.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(arg688_1.data_ptr()), c_void_p(arg689_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf841.data_ptr()))
    del arg324_1
    del arg325_1
    del arg688_1
    del arg689_1
    del buf777
    del buf778
    del buf779
    del buf780
    del buf781
    del buf782
    del buf783
    del buf784
    del buf785
    del buf786
    del buf787
    # Source Nodes: [bottleneck_output_104, l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf790 = extern_kernels.convolution(buf789, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf790, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg326_1
    del buf789
    buf791 = buf790; del buf790  # reuse
    buf792 = buf775; del buf775  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_109(c_void_p(buf791.data_ptr()), c_void_p(arg691_1.data_ptr()), c_void_p(arg692_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(buf792.data_ptr()))
    del arg327_1
    del arg328_1
    del arg329_1
    del arg691_1
    del arg692_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer11_norm2, l__mod___features_denseblock4_denselayer11_relu2, new_features_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf793 = extern_kernels.convolution(buf791, buf792, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf793, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf791
    buf800 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 672)  # alias
    buf818 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 672)  # alias
    buf837 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 672)  # alias
    buf866 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf857 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 672)  # alias
    buf805 = reinterpret_tensor(buf806, (4, 32, 7, 7), (42336, 1, 6048, 864), 832)  # alias
    buf823 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 832)  # alias
    buf842 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 832)  # alias
    buf862 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 832)  # alias
    buf807 = buf806; del buf806  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_110(c_void_p(buf807.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(arg694_1.data_ptr()), c_void_p(arg695_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(buf862.data_ptr()))
    del arg330_1
    del arg331_1
    del arg694_1
    del arg695_1
    del buf794
    del buf795
    del buf796
    del buf797
    del buf798
    del buf799
    del buf800
    del buf801
    del buf802
    del buf803
    del buf804
    del buf805
    # Source Nodes: [bottleneck_output_106, l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf808 = extern_kernels.convolution(buf807, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf808, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg332_1
    del buf807
    buf809 = buf808; del buf808  # reuse
    buf810 = buf792; del buf792  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_111(c_void_p(buf809.data_ptr()), c_void_p(arg697_1.data_ptr()), c_void_p(arg698_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(buf810.data_ptr()))
    del arg333_1
    del arg334_1
    del arg335_1
    del arg697_1
    del arg698_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer12_norm2, l__mod___features_denseblock4_denselayer12_relu2, new_features_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf811 = extern_kernels.convolution(buf809, buf810, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf811, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf809
    buf819 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 704)  # alias
    buf838 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 704)  # alias
    buf858 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 704)  # alias
    buf888 = empty_strided((4, 992, 7, 7), (48608, 1, 6944, 992), device='cpu', dtype=torch.float32)
    buf879 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 704)  # alias
    buf820 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 736)  # alias
    buf839 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 736)  # alias
    buf859 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 736)  # alias
    buf880 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 736)  # alias
    buf824 = reinterpret_tensor(buf825, (4, 32, 7, 7), (43904, 1, 6272, 896), 864)  # alias
    buf843 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 864)  # alias
    buf863 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 864)  # alias
    buf884 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 864)  # alias
    buf826 = buf825; del buf825  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_112(c_void_p(buf826.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(arg700_1.data_ptr()), c_void_p(arg701_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf858.data_ptr()), c_void_p(buf879.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf859.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf863.data_ptr()), c_void_p(buf884.data_ptr()))
    del arg336_1
    del arg337_1
    del arg700_1
    del arg701_1
    del buf812
    del buf813
    del buf814
    del buf815
    del buf816
    del buf817
    del buf818
    del buf819
    del buf820
    del buf821
    del buf822
    del buf823
    del buf824
    # Source Nodes: [bottleneck_output_108, l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf827 = extern_kernels.convolution(buf826, arg338_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf827, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg338_1
    del buf826
    buf828 = buf827; del buf827  # reuse
    buf829 = buf810; del buf810  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_113(c_void_p(buf828.data_ptr()), c_void_p(arg703_1.data_ptr()), c_void_p(arg704_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(buf829.data_ptr()))
    del arg339_1
    del arg340_1
    del arg341_1
    del arg703_1
    del arg704_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer13_norm2, l__mod___features_denseblock4_denselayer13_relu2, new_features_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf830 = extern_kernels.convolution(buf828, buf829, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf830, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf828
    buf832 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 512)  # alias
    buf852 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 512)  # alias
    buf873 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 512)  # alias
    buf911 = reinterpret_tensor(buf212, (4, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf212  # reuse
    buf895 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 512)  # alias
    buf833 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 544)  # alias
    buf853 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 544)  # alias
    buf874 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 544)  # alias
    buf896 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 544)  # alias
    buf834 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 576)  # alias
    buf854 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 576)  # alias
    buf875 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 576)  # alias
    buf897 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 576)  # alias
    buf835 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 608)  # alias
    buf855 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 608)  # alias
    buf876 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 608)  # alias
    buf898 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 608)  # alias
    buf840 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 768)  # alias
    buf860 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 768)  # alias
    buf881 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 768)  # alias
    buf903 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 768)  # alias
    buf844 = reinterpret_tensor(buf845, (4, 32, 7, 7), (45472, 1, 6496, 928), 896)  # alias
    buf864 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 896)  # alias
    buf885 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 896)  # alias
    buf907 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 896)  # alias
    buf846 = buf845; del buf845  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_114(c_void_p(buf846.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(arg706_1.data_ptr()), c_void_p(arg707_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf873.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf897.data_ptr()), c_void_p(buf835.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf876.data_ptr()), c_void_p(buf898.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf860.data_ptr()), c_void_p(buf881.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf907.data_ptr()))
    del arg342_1
    del arg343_1
    del arg706_1
    del arg707_1
    del buf677
    del buf683
    del buf689
    del buf695
    del buf760
    del buf830
    del buf831
    del buf832
    del buf833
    del buf834
    del buf835
    del buf836
    del buf837
    del buf838
    del buf839
    del buf840
    del buf841
    del buf842
    del buf843
    del buf844
    # Source Nodes: [bottleneck_output_110, l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf847 = extern_kernels.convolution(buf846, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf847, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg344_1
    del buf846
    buf848 = buf847; del buf847  # reuse
    buf849 = buf829; del buf829  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_115(c_void_p(buf848.data_ptr()), c_void_p(arg709_1.data_ptr()), c_void_p(arg710_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(buf849.data_ptr()))
    del arg345_1
    del arg346_1
    del arg347_1
    del arg709_1
    del arg710_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer14_norm2, l__mod___features_denseblock4_denselayer14_relu2, new_features_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf850 = extern_kernels.convolution(buf848, buf849, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf850, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf848
    buf851 = reinterpret_tensor(buf866, (4, 512, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    buf872 = reinterpret_tensor(buf888, (4, 512, 7, 7), (48608, 1, 6944, 992), 0)  # alias
    buf894 = reinterpret_tensor(buf911, (4, 512, 7, 7), (50176, 1, 7168, 1024), 0)  # alias
    buf856 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 640)  # alias
    buf877 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 640)  # alias
    buf899 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 640)  # alias
    buf861 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 800)  # alias
    buf882 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 800)  # alias
    buf904 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 800)  # alias
    buf865 = reinterpret_tensor(buf866, (4, 32, 7, 7), (47040, 1, 6720, 960), 928)  # alias
    buf886 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 928)  # alias
    buf908 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 928)  # alias
    buf867 = buf866; del buf866  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_116(c_void_p(buf867.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(arg712_1.data_ptr()), c_void_p(arg713_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf877.data_ptr()), c_void_p(buf899.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(buf882.data_ptr()), c_void_p(buf904.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf908.data_ptr()))
    del arg348_1
    del arg349_1
    del arg712_1
    del arg713_1
    del buf696
    del buf701
    del buf706
    del buf776
    del buf850
    del buf851
    del buf852
    del buf853
    del buf854
    del buf855
    del buf856
    del buf857
    del buf858
    del buf859
    del buf860
    del buf861
    del buf862
    del buf863
    del buf864
    del buf865
    # Source Nodes: [bottleneck_output_112, l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf868 = extern_kernels.convolution(buf867, arg350_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf868, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg350_1
    del buf867
    buf869 = buf868; del buf868  # reuse
    buf870 = buf849; del buf849  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_117(c_void_p(buf869.data_ptr()), c_void_p(arg715_1.data_ptr()), c_void_p(arg716_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(buf870.data_ptr()))
    del arg351_1
    del arg352_1
    del arg353_1
    del arg715_1
    del arg716_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer15_norm2, l__mod___features_denseblock4_denselayer15_relu2, new_features_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf871 = extern_kernels.convolution(buf869, buf870, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf871, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf869
    buf878 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 672)  # alias
    buf900 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 672)  # alias
    buf883 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 832)  # alias
    buf905 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 832)  # alias
    buf887 = reinterpret_tensor(buf888, (4, 32, 7, 7), (48608, 1, 6944, 992), 960)  # alias
    buf909 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 960)  # alias
    buf889 = buf888; del buf888  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_118(c_void_p(buf889.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(arg718_1.data_ptr()), c_void_p(arg719_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(buf905.data_ptr()), c_void_p(buf887.data_ptr()), c_void_p(buf909.data_ptr()))
    del arg354_1
    del arg355_1
    del arg718_1
    del arg719_1
    del buf718
    del buf793
    del buf871
    del buf872
    del buf873
    del buf874
    del buf875
    del buf876
    del buf877
    del buf878
    del buf879
    del buf880
    del buf881
    del buf882
    del buf883
    del buf884
    del buf885
    del buf886
    del buf887
    # Source Nodes: [bottleneck_output_114, l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf890 = extern_kernels.convolution(buf889, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf890, (4, 128, 7, 7), (6272, 1, 896, 128))
    del arg356_1
    del buf889
    buf891 = buf890; del buf890  # reuse
    buf892 = buf870; del buf870  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_119(c_void_p(buf891.data_ptr()), c_void_p(arg721_1.data_ptr()), c_void_p(arg722_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(buf892.data_ptr()))
    del arg357_1
    del arg358_1
    del arg359_1
    del arg721_1
    del arg722_1
    # Source Nodes: [l__mod___features_denseblock4_denselayer16_norm2, l__mod___features_denseblock4_denselayer16_relu2, new_features_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf893 = extern_kernels.convolution(buf891, buf892, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf893, (4, 32, 7, 7), (1568, 1, 224, 32))
    del buf891
    del buf892
    buf901 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 704)  # alias
    buf902 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 736)  # alias
    buf906 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 864)  # alias
    buf910 = reinterpret_tensor(buf911, (4, 32, 7, 7), (50176, 1, 7168, 1024), 992)  # alias
    buf912 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf913 = reinterpret_tensor(buf912, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf912  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_120(c_void_p(buf913.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(arg724_1.data_ptr()), c_void_p(arg725_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(buf901.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf910.data_ptr()))
    del arg360_1
    del arg361_1
    del arg724_1
    del arg725_1
    del buf731
    del buf745
    del buf811
    del buf893
    del buf894
    del buf895
    del buf896
    del buf897
    del buf898
    del buf899
    del buf900
    del buf901
    del buf902
    del buf903
    del buf904
    del buf905
    del buf906
    del buf907
    del buf908
    del buf909
    del buf910
    del buf911
    buf914 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [out_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg363_1, reinterpret_tensor(buf913, (4, 1024), (1024, 1), 0), reinterpret_tensor(arg362_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf914)
    del arg362_1
    del arg363_1
    return (buf914, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg367_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg370_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg373_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg376_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg379_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg382_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg385_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg388_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg391_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg394_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg397_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg400_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg403_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg406_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg409_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg412_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg415_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg418_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg421_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg424_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg427_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg430_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg433_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg436_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg439_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg442_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg445_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg448_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg451_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg454_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg457_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg460_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg463_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg466_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg469_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg472_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg475_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg478_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg481_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg484_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg487_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg490_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg493_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg496_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg499_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg502_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg505_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg508_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg511_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg514_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg517_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg520_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg521_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg523_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg524_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg526_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg527_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg529_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg530_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg532_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg533_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg535_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg536_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg538_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg539_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg541_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg542_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg544_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg545_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg547_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg548_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg550_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg551_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg553_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg554_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg556_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg557_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg558_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg559_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg560_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg561_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg562_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg563_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg564_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg565_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg566_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg567_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg568_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg569_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg570_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg571_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg572_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg573_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg574_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg575_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg576_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg577_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg578_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg579_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg580_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg581_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg582_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg583_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg584_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg585_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg586_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg587_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg588_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg589_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg590_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg591_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg592_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg593_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg594_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg595_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg596_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg597_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg598_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg599_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg600_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg601_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg602_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg603_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg604_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg605_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg606_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg607_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg608_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg609_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg610_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg611_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg612_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg613_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg614_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg615_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg616_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg617_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg618_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg619_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg620_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg621_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg622_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg623_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg624_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg625_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg626_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg627_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg628_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg629_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg630_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg631_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg632_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg633_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg634_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg635_1 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    arg636_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg637_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg638_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg639_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg640_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg641_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg642_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg643_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg644_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg645_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg646_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg647_1 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    arg648_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg649_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg650_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg651_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg652_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg653_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg654_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg655_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg656_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg657_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg658_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg659_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg660_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg661_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg662_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg663_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg664_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg665_1 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    arg666_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg667_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg668_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg669_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg670_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg671_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg672_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg673_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg674_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg675_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg676_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg677_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg678_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg679_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg680_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg681_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg682_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg683_1 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    arg684_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg685_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg686_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg687_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg688_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg689_1 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    arg690_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg691_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg692_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg693_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg694_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg695_1 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    arg696_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg697_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg698_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg699_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg700_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg701_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg702_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg703_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg704_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg705_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg706_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg707_1 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    arg708_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg709_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg710_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg711_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg712_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg713_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg714_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg715_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg716_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg717_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg718_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg719_1 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    arg720_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg721_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg722_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg723_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg724_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg725_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg726_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg727_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('densenet121', benchmark_compiled_module)
