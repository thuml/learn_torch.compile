
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                            tmp69.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3584L*x1) + (200704L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_94 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_relu_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
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
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (512, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg73_1, (2048, ), (1, ))
    assert_size_stride(arg74_1, (2048, ), (1, ))
    assert_size_stride(arg75_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg76_1, (2048, ), (1, ))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg85_1, (2048, ), (1, ))
    assert_size_stride(arg86_1, (2048, ), (1, ))
    assert_size_stride(arg87_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (2048, ), (1, ))
    assert_size_stride(arg90_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg95_1, (2048, ), (1, ))
    assert_size_stride(arg96_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg97_1, (2048, ), (1, ))
    assert_size_stride(arg98_1, (2048, ), (1, ))
    assert_size_stride(arg99_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (2048, ), (1, ))
    assert_size_stride(arg104_1, (2048, ), (1, ))
    assert_size_stride(arg105_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg106_1, (2048, ), (1, ))
    assert_size_stride(arg107_1, (2048, ), (1, ))
    assert_size_stride(arg108_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg115_1, (2048, ), (1, ))
    assert_size_stride(arg116_1, (2048, ), (1, ))
    assert_size_stride(arg117_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg121_1, (2048, ), (1, ))
    assert_size_stride(arg122_1, (2048, ), (1, ))
    assert_size_stride(arg123_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg124_1, (2048, ), (1, ))
    assert_size_stride(arg125_1, (2048, ), (1, ))
    assert_size_stride(arg126_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg130_1, (2048, ), (1, ))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (2048, ), (1, ))
    assert_size_stride(arg135_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg142_1, (2048, ), (1, ))
    assert_size_stride(arg143_1, (2048, ), (1, ))
    assert_size_stride(arg144_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (2048, ), (1, ))
    assert_size_stride(arg153_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg160_1, (2048, ), (1, ))
    assert_size_stride(arg161_1, (2048, ), (1, ))
    assert_size_stride(arg162_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg166_1, (2048, ), (1, ))
    assert_size_stride(arg167_1, (2048, ), (1, ))
    assert_size_stride(arg168_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (2048, ), (1, ))
    assert_size_stride(arg177_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg178_1, (2048, ), (1, ))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (2048, ), (1, ))
    assert_size_stride(arg186_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (2048, ), (1, ))
    assert_size_stride(arg189_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg193_1, (2048, ), (1, ))
    assert_size_stride(arg194_1, (2048, ), (1, ))
    assert_size_stride(arg195_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (2048, ), (1, ))
    assert_size_stride(arg198_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg202_1, (2048, ), (1, ))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (2048, ), (1, ))
    assert_size_stride(arg207_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (2048, ), (1, ))
    assert_size_stride(arg213_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg214_1, (2048, ), (1, ))
    assert_size_stride(arg215_1, (2048, ), (1, ))
    assert_size_stride(arg216_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, ), (1, ))
    assert_size_stride(arg222_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg232_1, (2048, ), (1, ))
    assert_size_stride(arg233_1, (2048, ), (1, ))
    assert_size_stride(arg234_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg238_1, (2048, ), (1, ))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg241_1, (2048, ), (1, ))
    assert_size_stride(arg242_1, (2048, ), (1, ))
    assert_size_stride(arg243_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, ), (1, ))
    assert_size_stride(arg252_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg256_1, (2048, ), (1, ))
    assert_size_stride(arg257_1, (2048, ), (1, ))
    assert_size_stride(arg258_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg259_1, (2048, ), (1, ))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (2048, ), (1, ))
    assert_size_stride(arg267_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg274_1, (2048, ), (1, ))
    assert_size_stride(arg275_1, (2048, ), (1, ))
    assert_size_stride(arg276_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg277_1, (2048, ), (1, ))
    assert_size_stride(arg278_1, (2048, ), (1, ))
    assert_size_stride(arg279_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg283_1, (4096, ), (1, ))
    assert_size_stride(arg284_1, (4096, ), (1, ))
    assert_size_stride(arg285_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg286_1, (4096, ), (1, ))
    assert_size_stride(arg287_1, (4096, ), (1, ))
    assert_size_stride(arg288_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg289_1, (2048, ), (1, ))
    assert_size_stride(arg290_1, (2048, ), (1, ))
    assert_size_stride(arg291_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg292_1, (2048, ), (1, ))
    assert_size_stride(arg293_1, (2048, ), (1, ))
    assert_size_stride(arg294_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg295_1, (4096, ), (1, ))
    assert_size_stride(arg296_1, (4096, ), (1, ))
    assert_size_stride(arg297_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg298_1, (4096, ), (1, ))
    assert_size_stride(arg299_1, (4096, ), (1, ))
    assert_size_stride(arg300_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (2048, ), (1, ))
    assert_size_stride(arg303_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg304_1, (4096, ), (1, ))
    assert_size_stride(arg305_1, (4096, ), (1, ))
    assert_size_stride(arg306_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg307_1, (4096, ), (1, ))
    assert_size_stride(arg308_1, (4096, ), (1, ))
    assert_size_stride(arg309_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg310_1, (2048, ), (1, ))
    assert_size_stride(arg311_1, (2048, ), (1, ))
    assert_size_stride(arg312_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg313_1, (1000, ), (1, ))
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (256, ), (1, ))
    assert_size_stride(arg324_1, (256, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (256, ), (1, ))
    assert_size_stride(arg327_1, (256, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (256, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (), ())
    assert_size_stride(arg341_1, (512, ), (1, ))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (), ())
    assert_size_stride(arg344_1, (256, ), (1, ))
    assert_size_stride(arg345_1, (256, ), (1, ))
    assert_size_stride(arg346_1, (), ())
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (), ())
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (), ())
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (512, ), (1, ))
    assert_size_stride(arg355_1, (), ())
    assert_size_stride(arg356_1, (512, ), (1, ))
    assert_size_stride(arg357_1, (512, ), (1, ))
    assert_size_stride(arg358_1, (), ())
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (), ())
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (), ())
    assert_size_stride(arg365_1, (512, ), (1, ))
    assert_size_stride(arg366_1, (512, ), (1, ))
    assert_size_stride(arg367_1, (), ())
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (), ())
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (), ())
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (512, ), (1, ))
    assert_size_stride(arg376_1, (), ())
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (), ())
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (), ())
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (), ())
    assert_size_stride(arg386_1, (2048, ), (1, ))
    assert_size_stride(arg387_1, (2048, ), (1, ))
    assert_size_stride(arg388_1, (), ())
    assert_size_stride(arg389_1, (2048, ), (1, ))
    assert_size_stride(arg390_1, (2048, ), (1, ))
    assert_size_stride(arg391_1, (), ())
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (), ())
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, ), (1, ))
    assert_size_stride(arg397_1, (), ())
    assert_size_stride(arg398_1, (2048, ), (1, ))
    assert_size_stride(arg399_1, (2048, ), (1, ))
    assert_size_stride(arg400_1, (), ())
    assert_size_stride(arg401_1, (2048, ), (1, ))
    assert_size_stride(arg402_1, (2048, ), (1, ))
    assert_size_stride(arg403_1, (), ())
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (), ())
    assert_size_stride(arg407_1, (2048, ), (1, ))
    assert_size_stride(arg408_1, (2048, ), (1, ))
    assert_size_stride(arg409_1, (), ())
    assert_size_stride(arg410_1, (2048, ), (1, ))
    assert_size_stride(arg411_1, (2048, ), (1, ))
    assert_size_stride(arg412_1, (), ())
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (), ())
    assert_size_stride(arg416_1, (2048, ), (1, ))
    assert_size_stride(arg417_1, (2048, ), (1, ))
    assert_size_stride(arg418_1, (), ())
    assert_size_stride(arg419_1, (2048, ), (1, ))
    assert_size_stride(arg420_1, (2048, ), (1, ))
    assert_size_stride(arg421_1, (), ())
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (), ())
    assert_size_stride(arg425_1, (2048, ), (1, ))
    assert_size_stride(arg426_1, (2048, ), (1, ))
    assert_size_stride(arg427_1, (), ())
    assert_size_stride(arg428_1, (2048, ), (1, ))
    assert_size_stride(arg429_1, (2048, ), (1, ))
    assert_size_stride(arg430_1, (), ())
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (), ())
    assert_size_stride(arg434_1, (2048, ), (1, ))
    assert_size_stride(arg435_1, (2048, ), (1, ))
    assert_size_stride(arg436_1, (), ())
    assert_size_stride(arg437_1, (2048, ), (1, ))
    assert_size_stride(arg438_1, (2048, ), (1, ))
    assert_size_stride(arg439_1, (), ())
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (), ())
    assert_size_stride(arg443_1, (2048, ), (1, ))
    assert_size_stride(arg444_1, (2048, ), (1, ))
    assert_size_stride(arg445_1, (), ())
    assert_size_stride(arg446_1, (2048, ), (1, ))
    assert_size_stride(arg447_1, (2048, ), (1, ))
    assert_size_stride(arg448_1, (), ())
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (2048, ), (1, ))
    assert_size_stride(arg453_1, (2048, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (2048, ), (1, ))
    assert_size_stride(arg456_1, (2048, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (1024, ), (1, ))
    assert_size_stride(arg459_1, (1024, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (2048, ), (1, ))
    assert_size_stride(arg462_1, (2048, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (2048, ), (1, ))
    assert_size_stride(arg465_1, (2048, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (2048, ), (1, ))
    assert_size_stride(arg471_1, (2048, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (2048, ), (1, ))
    assert_size_stride(arg474_1, (2048, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (1024, ), (1, ))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (2048, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (2048, ), (1, ))
    assert_size_stride(arg483_1, (2048, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (1024, ), (1, ))
    assert_size_stride(arg486_1, (1024, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (2048, ), (1, ))
    assert_size_stride(arg489_1, (2048, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (2048, ), (1, ))
    assert_size_stride(arg492_1, (2048, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (2048, ), (1, ))
    assert_size_stride(arg498_1, (2048, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (2048, ), (1, ))
    assert_size_stride(arg501_1, (2048, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (2048, ), (1, ))
    assert_size_stride(arg507_1, (2048, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (2048, ), (1, ))
    assert_size_stride(arg510_1, (2048, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (2048, ), (1, ))
    assert_size_stride(arg516_1, (2048, ), (1, ))
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (2048, ), (1, ))
    assert_size_stride(arg519_1, (2048, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (1024, ), (1, ))
    assert_size_stride(arg522_1, (1024, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (2048, ), (1, ))
    assert_size_stride(arg525_1, (2048, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (2048, ), (1, ))
    assert_size_stride(arg528_1, (2048, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (1024, ), (1, ))
    assert_size_stride(arg531_1, (1024, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (2048, ), (1, ))
    assert_size_stride(arg534_1, (2048, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (2048, ), (1, ))
    assert_size_stride(arg537_1, (2048, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (1024, ), (1, ))
    assert_size_stride(arg540_1, (1024, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (2048, ), (1, ))
    assert_size_stride(arg543_1, (2048, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (2048, ), (1, ))
    assert_size_stride(arg546_1, (2048, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (1024, ), (1, ))
    assert_size_stride(arg549_1, (1024, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (2048, ), (1, ))
    assert_size_stride(arg552_1, (2048, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (2048, ), (1, ))
    assert_size_stride(arg555_1, (2048, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (1024, ), (1, ))
    assert_size_stride(arg558_1, (1024, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (2048, ), (1, ))
    assert_size_stride(arg561_1, (2048, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (2048, ), (1, ))
    assert_size_stride(arg564_1, (2048, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (1024, ), (1, ))
    assert_size_stride(arg567_1, (1024, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (2048, ), (1, ))
    assert_size_stride(arg570_1, (2048, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (2048, ), (1, ))
    assert_size_stride(arg573_1, (2048, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (1024, ), (1, ))
    assert_size_stride(arg576_1, (1024, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (2048, ), (1, ))
    assert_size_stride(arg579_1, (2048, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (2048, ), (1, ))
    assert_size_stride(arg582_1, (2048, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (1024, ), (1, ))
    assert_size_stride(arg585_1, (1024, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (2048, ), (1, ))
    assert_size_stride(arg588_1, (2048, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (2048, ), (1, ))
    assert_size_stride(arg591_1, (2048, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (1024, ), (1, ))
    assert_size_stride(arg594_1, (1024, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (4096, ), (1, ))
    assert_size_stride(arg597_1, (4096, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (4096, ), (1, ))
    assert_size_stride(arg600_1, (4096, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (2048, ), (1, ))
    assert_size_stride(arg603_1, (2048, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (2048, ), (1, ))
    assert_size_stride(arg606_1, (2048, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (4096, ), (1, ))
    assert_size_stride(arg609_1, (4096, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (4096, ), (1, ))
    assert_size_stride(arg612_1, (4096, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (2048, ), (1, ))
    assert_size_stride(arg615_1, (2048, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (4096, ), (1, ))
    assert_size_stride(arg618_1, (4096, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (4096, ), (1, ))
    assert_size_stride(arg621_1, (4096, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (2048, ), (1, ))
    assert_size_stride(arg624_1, (2048, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg626_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg626_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg1_1
    del arg2_1
    del arg314_1
    del arg315_1
    del buf3
    # Source Nodes: [x_4], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del arg3_1
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg317_1
    del arg318_1
    del arg4_1
    del arg5_1
    del arg6_1
    # Source Nodes: [x_5, x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf8, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del buf6
    buf9 = buf8; del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()))
    del arg320_1
    del arg321_1
    del arg7_1
    del arg8_1
    # Source Nodes: [x_10, x_12, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf10 = extern_kernels.convolution(buf9, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg9_1
    del buf9
    # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf4, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg12_1
    del buf4
    buf12 = buf10; del buf10  # reuse
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_4(c_void_p(buf13.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg13_1
    del arg14_1
    del arg323_1
    del arg324_1
    del arg326_1
    del arg327_1
    del buf11
    # Source Nodes: [shortcut_2, x_16], Original ATen: [aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf13, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del arg15_1
    buf15 = buf14; del buf14  # reuse
    buf16 = buf7; del buf7  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5(c_void_p(buf15.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg16_1
    del arg17_1
    del arg18_1
    del arg329_1
    del arg330_1
    # Source Nodes: [x_17, x_18, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf17, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del buf15
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf18.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()))
    del arg19_1
    del arg20_1
    del arg332_1
    del arg333_1
    # Source Nodes: [x_20, x_22, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf18, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg21_1
    del buf18
    buf20 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_7(c_void_p(buf20.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg335_1
    del arg336_1
    del buf19
    # Source Nodes: [x_28], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del arg24_1
    buf22 = buf21; del buf21  # reuse
    buf23 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8(c_void_p(buf22.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg25_1
    del arg26_1
    del arg27_1
    del arg338_1
    del arg339_1
    # Source Nodes: [x_29, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf24, (8, 512, 56, 56), (1605632, 1, 28672, 512))
    del buf22
    del buf23
    buf25 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf25.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg28_1
    del arg29_1
    del arg341_1
    del arg342_1
    # Source Nodes: [x_32, x_34, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf25, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg30_1
    del buf25
    buf27 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_10(c_void_p(buf27.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()))
    del arg31_1
    del arg32_1
    del arg344_1
    del arg345_1
    del buf26
    # Source Nodes: [x_41], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 1024, 56, 56), (3211264, 1, 57344, 1024))
    del arg33_1
    buf29 = buf28; del buf28  # reuse
    buf30 = empty_strided((1024, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11(c_void_p(buf29.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg347_1
    del arg348_1
    del arg34_1
    del arg35_1
    del arg36_1
    # Source Nodes: [x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf31 = extern_kernels.convolution(buf29, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf31, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del buf29
    buf32 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf32.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()))
    del arg350_1
    del arg351_1
    del arg37_1
    del arg38_1
    # Source Nodes: [x_45, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf33 = extern_kernels.convolution(buf32, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg39_1
    del buf32
    # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf27, arg42_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg42_1
    del buf27
    buf35 = buf33; del buf33  # reuse
    buf36 = buf35; del buf35  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_13(c_void_p(buf36.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()))
    del arg353_1
    del arg354_1
    del arg356_1
    del arg357_1
    del arg40_1
    del arg41_1
    del arg43_1
    del arg44_1
    del buf34
    # Source Nodes: [shortcut_6, x_53], Original ATen: [aten.convolution, aten.relu]
    buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del arg45_1
    buf38 = buf37; del buf37  # reuse
    buf39 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14(c_void_p(buf38.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf39.data_ptr()))
    del arg359_1
    del arg360_1
    del arg46_1
    del arg47_1
    del arg48_1
    # Source Nodes: [x_54, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf40, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del buf38
    buf41 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf41.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()))
    del arg362_1
    del arg363_1
    del arg49_1
    del arg50_1
    # Source Nodes: [x_57, x_59, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf42 = extern_kernels.convolution(buf41, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg51_1
    del buf41
    buf43 = buf36; del buf36  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf43.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg365_1
    del arg366_1
    del arg52_1
    del arg53_1
    del buf42
    # Source Nodes: [x_65], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del arg54_1
    buf45 = buf44; del buf44  # reuse
    buf46 = buf39; del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf45.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg368_1
    del arg369_1
    del arg55_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_66, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf47, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del buf45
    buf48 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf48.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg371_1
    del arg372_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_69, x_71, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf49 = extern_kernels.convolution(buf48, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg60_1
    del buf48
    buf50 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_19(c_void_p(buf50.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()))
    del arg374_1
    del arg375_1
    del arg61_1
    del arg62_1
    del buf49
    # Source Nodes: [x_77], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf50, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del arg63_1
    buf52 = buf51; del buf51  # reuse
    buf53 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20(c_void_p(buf52.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg377_1
    del arg378_1
    del arg64_1
    del arg65_1
    del arg66_1
    # Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf54, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    del buf52
    del buf53
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf55.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()))
    del arg380_1
    del arg381_1
    del arg67_1
    del arg68_1
    # Source Nodes: [x_81, x_83, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf56 = extern_kernels.convolution(buf55, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg69_1
    del buf55
    buf57 = buf50; del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_22(c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg383_1
    del arg384_1
    del arg70_1
    del arg71_1
    del buf56
    # Source Nodes: [x_90], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (8, 2048, 28, 28), (1605632, 1, 57344, 2048))
    del arg72_1
    buf59 = buf58; del buf58  # reuse
    buf60 = empty_strided((2048, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23(c_void_p(buf59.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg386_1
    del arg387_1
    del arg73_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_91, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf61 = extern_kernels.convolution(buf59, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf61, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf59
    buf62 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf62.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg389_1
    del arg390_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_94, x_96, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf63 = extern_kernels.convolution(buf62, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg78_1
    del buf62
    # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf57, arg81_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg81_1
    del buf57
    buf65 = buf63; del buf63  # reuse
    buf66 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_25(c_void_p(buf66.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg392_1
    del arg393_1
    del arg395_1
    del arg396_1
    del arg79_1
    del arg80_1
    del arg82_1
    del arg83_1
    del buf64
    # Source Nodes: [shortcut_11, x_102], Original ATen: [aten.convolution, aten.relu]
    buf67 = extern_kernels.convolution(buf66, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg84_1
    buf68 = buf67; del buf67  # reuse
    buf69 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26(c_void_p(buf68.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg398_1
    del arg399_1
    del arg85_1
    del arg86_1
    del arg87_1
    # Source Nodes: [x_103, x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf70, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf68
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf71.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg401_1
    del arg402_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_106, x_108, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf72 = extern_kernels.convolution(buf71, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf72, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg90_1
    del buf71
    buf73 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_28(c_void_p(buf73.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()))
    del arg404_1
    del arg405_1
    del arg91_1
    del arg92_1
    del buf72
    # Source Nodes: [x_114], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg93_1
    buf75 = buf74; del buf74  # reuse
    buf76 = buf69; del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29(c_void_p(buf75.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg407_1
    del arg408_1
    del arg94_1
    del arg95_1
    del arg96_1
    # Source Nodes: [x_115, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf77, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf75
    buf78 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf78.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()))
    del arg410_1
    del arg411_1
    del arg97_1
    del arg98_1
    # Source Nodes: [x_118, x_120, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf79 = extern_kernels.convolution(buf78, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg99_1
    del buf78
    buf80 = buf73; del buf73  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_31(c_void_p(buf80.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg413_1
    del arg414_1
    del buf79
    # Source Nodes: [x_126], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg102_1
    buf82 = buf81; del buf81  # reuse
    buf83 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf82.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg103_1
    del arg104_1
    del arg105_1
    del arg416_1
    del arg417_1
    # Source Nodes: [x_127, x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf84, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf82
    buf85 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf85.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg419_1
    del arg420_1
    # Source Nodes: [x_130, x_132, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf86 = extern_kernels.convolution(buf85, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg108_1
    del buf85
    buf87 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_34(c_void_p(buf87.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg423_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg422_1
    del arg423_1
    del buf86
    # Source Nodes: [x_138], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg111_1
    buf89 = buf88; del buf88  # reuse
    buf90 = buf83; del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35(c_void_p(buf89.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg112_1
    del arg113_1
    del arg114_1
    del arg425_1
    del arg426_1
    # Source Nodes: [x_139, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf91 = extern_kernels.convolution(buf89, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf91, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf89
    buf92 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf92.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg429_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg428_1
    del arg429_1
    # Source Nodes: [x_142, x_144, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf93 = extern_kernels.convolution(buf92, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf93, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg117_1
    del buf92
    buf94 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_37(c_void_p(buf94.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg431_1
    del arg432_1
    del buf93
    # Source Nodes: [x_150], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg120_1
    buf96 = buf95; del buf95  # reuse
    buf97 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_38(c_void_p(buf96.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg435_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg121_1
    del arg122_1
    del arg123_1
    del arg434_1
    del arg435_1
    # Source Nodes: [x_151, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf98, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf96
    buf99 = buf98; del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf99.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg437_1
    del arg438_1
    # Source Nodes: [x_154, x_156, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf100 = extern_kernels.convolution(buf99, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg126_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_40(c_void_p(buf101.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg127_1
    del arg128_1
    del arg440_1
    del arg441_1
    del buf94
    # Source Nodes: [x_162], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg129_1
    buf103 = buf102; del buf102  # reuse
    buf104 = buf97; del buf97  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41(c_void_p(buf103.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf104.data_ptr()))
    del arg130_1
    del arg131_1
    del arg132_1
    del arg443_1
    del arg444_1
    # Source Nodes: [x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf105, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf103
    buf106 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf106.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg446_1
    del arg447_1
    # Source Nodes: [x_166, x_168, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf107 = extern_kernels.convolution(buf106, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg135_1
    del buf106
    buf108 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_43(c_void_p(buf108.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg136_1
    del arg137_1
    del arg449_1
    del arg450_1
    del buf107
    # Source Nodes: [x_174], Original ATen: [aten.convolution]
    buf109 = extern_kernels.convolution(buf108, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf109, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg138_1
    buf110 = buf109; del buf109  # reuse
    buf111 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44(c_void_p(buf110.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg139_1
    del arg140_1
    del arg141_1
    del arg452_1
    del arg453_1
    # Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf112, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf110
    buf113 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_45(c_void_p(buf113.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg455_1
    del arg456_1
    # Source Nodes: [x_178, x_180, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf114 = extern_kernels.convolution(buf113, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg144_1
    del buf113
    buf115 = buf108; del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf115.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg459_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg458_1
    del arg459_1
    del buf114
    # Source Nodes: [x_186], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg147_1
    buf117 = buf116; del buf116  # reuse
    buf118 = buf111; del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47(c_void_p(buf117.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg148_1
    del arg149_1
    del arg150_1
    del arg461_1
    del arg462_1
    # Source Nodes: [x_187, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf119, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf117
    buf120 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf120.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg465_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg464_1
    del arg465_1
    # Source Nodes: [x_190, x_192, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf121 = extern_kernels.convolution(buf120, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf121, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg153_1
    del buf120
    buf122 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_49(c_void_p(buf122.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()))
    del arg154_1
    del arg155_1
    del arg467_1
    del arg468_1
    del buf121
    # Source Nodes: [x_198], Original ATen: [aten.convolution]
    buf123 = extern_kernels.convolution(buf122, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf123, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg156_1
    buf124 = buf123; del buf123  # reuse
    buf125 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50(c_void_p(buf124.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf125.data_ptr()))
    del arg157_1
    del arg158_1
    del arg159_1
    del arg470_1
    del arg471_1
    # Source Nodes: [x_199, x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf124, buf125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf126, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf124
    buf127 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_51(c_void_p(buf127.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg474_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()))
    del arg160_1
    del arg161_1
    del arg473_1
    del arg474_1
    # Source Nodes: [x_202, x_204, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf128 = extern_kernels.convolution(buf127, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg162_1
    del buf127
    buf129 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_52(c_void_p(buf129.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg477_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()))
    del arg163_1
    del arg164_1
    del arg476_1
    del arg477_1
    del buf128
    # Source Nodes: [x_210], Original ATen: [aten.convolution]
    buf130 = extern_kernels.convolution(buf129, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg165_1
    buf131 = buf130; del buf130  # reuse
    buf132 = buf125; del buf125  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53(c_void_p(buf131.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf132.data_ptr()))
    del arg166_1
    del arg167_1
    del arg168_1
    del arg479_1
    del arg480_1
    # Source Nodes: [x_211, x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf133 = extern_kernels.convolution(buf131, buf132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf133, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf131
    buf134 = buf133; del buf133  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_54(c_void_p(buf134.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg483_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()))
    del arg169_1
    del arg170_1
    del arg482_1
    del arg483_1
    # Source Nodes: [x_214, x_216, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf135 = extern_kernels.convolution(buf134, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf135, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg171_1
    del buf134
    buf136 = buf129; del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_55(c_void_p(buf136.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg486_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()))
    del arg172_1
    del arg173_1
    del arg485_1
    del arg486_1
    del buf135
    # Source Nodes: [x_222], Original ATen: [aten.convolution]
    buf137 = extern_kernels.convolution(buf136, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf137, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg174_1
    buf138 = buf137; del buf137  # reuse
    buf139 = buf132; del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56(c_void_p(buf138.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf139.data_ptr()))
    del arg175_1
    del arg176_1
    del arg177_1
    del arg488_1
    del arg489_1
    # Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf140, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf138
    buf141 = buf140; del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_57(c_void_p(buf141.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg492_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()))
    del arg178_1
    del arg179_1
    del arg491_1
    del arg492_1
    # Source Nodes: [x_226, x_228, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf142 = extern_kernels.convolution(buf141, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg180_1
    del buf141
    buf143 = buf136; del buf136  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_58(c_void_p(buf143.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg495_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()))
    del arg181_1
    del arg182_1
    del arg494_1
    del arg495_1
    del buf142
    # Source Nodes: [x_234], Original ATen: [aten.convolution]
    buf144 = extern_kernels.convolution(buf143, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf144, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg183_1
    buf145 = buf144; del buf144  # reuse
    buf146 = buf139; del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59(c_void_p(buf145.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg498_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg184_1
    del arg185_1
    del arg186_1
    del arg497_1
    del arg498_1
    # Source Nodes: [x_235, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf147, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf145
    buf148 = buf147; del buf147  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_60(c_void_p(buf148.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg501_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()))
    del arg187_1
    del arg188_1
    del arg500_1
    del arg501_1
    # Source Nodes: [x_238, x_240, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf149 = extern_kernels.convolution(buf148, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg189_1
    del buf148
    buf150 = buf143; del buf143  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_61(c_void_p(buf150.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg504_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()))
    del arg190_1
    del arg191_1
    del arg503_1
    del arg504_1
    del buf149
    # Source Nodes: [x_246], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(buf150, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg192_1
    buf152 = buf151; del buf151  # reuse
    buf153 = buf146; del buf146  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf152.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg507_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg193_1
    del arg194_1
    del arg195_1
    del arg506_1
    del arg507_1
    # Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf154, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf152
    buf155 = buf154; del buf154  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_63(c_void_p(buf155.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg510_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()))
    del arg196_1
    del arg197_1
    del arg509_1
    del arg510_1
    # Source Nodes: [x_250, x_252, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf156 = extern_kernels.convolution(buf155, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf156, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg198_1
    del buf155
    buf157 = buf150; del buf150  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_64(c_void_p(buf157.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(arg512_1.data_ptr()), c_void_p(arg513_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()))
    del arg199_1
    del arg200_1
    del arg512_1
    del arg513_1
    del buf156
    # Source Nodes: [x_258], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg201_1
    buf159 = buf158; del buf158  # reuse
    buf160 = buf153; del buf153  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65(c_void_p(buf159.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(arg516_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg202_1
    del arg203_1
    del arg204_1
    del arg515_1
    del arg516_1
    # Source Nodes: [x_259, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf161, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf159
    buf162 = buf161; del buf161  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_66(c_void_p(buf162.data_ptr()), c_void_p(arg518_1.data_ptr()), c_void_p(arg519_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()))
    del arg205_1
    del arg206_1
    del arg518_1
    del arg519_1
    # Source Nodes: [x_262, x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf163 = extern_kernels.convolution(buf162, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf163, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg207_1
    del buf162
    buf164 = buf157; del buf157  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_67(c_void_p(buf164.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg521_1.data_ptr()), c_void_p(arg522_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()))
    del arg208_1
    del arg209_1
    del arg521_1
    del arg522_1
    del buf163
    # Source Nodes: [x_270], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(buf164, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf165, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg210_1
    buf166 = buf165; del buf165  # reuse
    buf167 = buf160; del buf160  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68(c_void_p(buf166.data_ptr()), c_void_p(arg524_1.data_ptr()), c_void_p(arg525_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg211_1
    del arg212_1
    del arg213_1
    del arg524_1
    del arg525_1
    # Source Nodes: [x_271, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf168 = extern_kernels.convolution(buf166, buf167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf168, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf166
    buf169 = buf168; del buf168  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_69(c_void_p(buf169.data_ptr()), c_void_p(arg527_1.data_ptr()), c_void_p(arg528_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg527_1
    del arg528_1
    # Source Nodes: [x_274, x_276, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf170 = extern_kernels.convolution(buf169, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg216_1
    del buf169
    buf171 = buf164; del buf164  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_70(c_void_p(buf171.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(arg530_1.data_ptr()), c_void_p(arg531_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()))
    del arg217_1
    del arg218_1
    del arg530_1
    del arg531_1
    del buf170
    # Source Nodes: [x_282], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg219_1
    buf173 = buf172; del buf172  # reuse
    buf174 = buf167; del buf167  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_71(c_void_p(buf173.data_ptr()), c_void_p(arg533_1.data_ptr()), c_void_p(arg534_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg220_1
    del arg221_1
    del arg222_1
    del arg533_1
    del arg534_1
    # Source Nodes: [x_283, x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf175, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf173
    buf176 = buf175; del buf175  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_72(c_void_p(buf176.data_ptr()), c_void_p(arg536_1.data_ptr()), c_void_p(arg537_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()))
    del arg223_1
    del arg224_1
    del arg536_1
    del arg537_1
    # Source Nodes: [x_286, x_288, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf177 = extern_kernels.convolution(buf176, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf177, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg225_1
    del buf176
    buf178 = buf171; del buf171  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_73(c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg539_1.data_ptr()), c_void_p(arg540_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg539_1
    del arg540_1
    del buf177
    # Source Nodes: [x_294], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf178, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf179, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg228_1
    buf180 = buf179; del buf179  # reuse
    buf181 = buf174; del buf174  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_74(c_void_p(buf180.data_ptr()), c_void_p(arg542_1.data_ptr()), c_void_p(arg543_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg229_1
    del arg230_1
    del arg231_1
    del arg542_1
    del arg543_1
    # Source Nodes: [x_295, x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf182, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf180
    buf183 = buf182; del buf182  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_75(c_void_p(buf183.data_ptr()), c_void_p(arg545_1.data_ptr()), c_void_p(arg546_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()))
    del arg232_1
    del arg233_1
    del arg545_1
    del arg546_1
    # Source Nodes: [x_298, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf184 = extern_kernels.convolution(buf183, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf184, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg234_1
    del buf183
    buf185 = buf178; del buf178  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_76(c_void_p(buf185.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(arg548_1.data_ptr()), c_void_p(arg549_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()))
    del arg235_1
    del arg236_1
    del arg548_1
    del arg549_1
    del buf184
    # Source Nodes: [x_306], Original ATen: [aten.convolution]
    buf186 = extern_kernels.convolution(buf185, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf186, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg237_1
    buf187 = buf186; del buf186  # reuse
    buf188 = buf181; del buf181  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77(c_void_p(buf187.data_ptr()), c_void_p(arg551_1.data_ptr()), c_void_p(arg552_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg238_1
    del arg239_1
    del arg240_1
    del arg551_1
    del arg552_1
    # Source Nodes: [x_307, x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf189, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf187
    buf190 = buf189; del buf189  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_78(c_void_p(buf190.data_ptr()), c_void_p(arg554_1.data_ptr()), c_void_p(arg555_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()))
    del arg241_1
    del arg242_1
    del arg554_1
    del arg555_1
    # Source Nodes: [x_310, x_312, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf191 = extern_kernels.convolution(buf190, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf191, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg243_1
    del buf190
    buf192 = buf185; del buf185  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_79(c_void_p(buf192.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(arg557_1.data_ptr()), c_void_p(arg558_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()))
    del arg244_1
    del arg245_1
    del arg557_1
    del arg558_1
    del buf191
    # Source Nodes: [x_318], Original ATen: [aten.convolution]
    buf193 = extern_kernels.convolution(buf192, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf193, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg246_1
    buf194 = buf193; del buf193  # reuse
    buf195 = buf188; del buf188  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_80(c_void_p(buf194.data_ptr()), c_void_p(arg560_1.data_ptr()), c_void_p(arg561_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg247_1
    del arg248_1
    del arg249_1
    del arg560_1
    del arg561_1
    # Source Nodes: [x_319, x_320, x_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf196, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf194
    buf197 = buf196; del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_81(c_void_p(buf197.data_ptr()), c_void_p(arg563_1.data_ptr()), c_void_p(arg564_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()))
    del arg250_1
    del arg251_1
    del arg563_1
    del arg564_1
    # Source Nodes: [x_322, x_324, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf198 = extern_kernels.convolution(buf197, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf198, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg252_1
    del buf197
    buf199 = buf192; del buf192  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_82(c_void_p(buf199.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(arg566_1.data_ptr()), c_void_p(arg567_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()))
    del arg253_1
    del arg254_1
    del arg566_1
    del arg567_1
    del buf198
    # Source Nodes: [x_330], Original ATen: [aten.convolution]
    buf200 = extern_kernels.convolution(buf199, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf200, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg255_1
    buf201 = buf200; del buf200  # reuse
    buf202 = buf195; del buf195  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_83(c_void_p(buf201.data_ptr()), c_void_p(arg569_1.data_ptr()), c_void_p(arg570_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf202.data_ptr()))
    del arg256_1
    del arg257_1
    del arg258_1
    del arg569_1
    del arg570_1
    # Source Nodes: [x_331, x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf203 = extern_kernels.convolution(buf201, buf202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf203, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf201
    buf204 = buf203; del buf203  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_84(c_void_p(buf204.data_ptr()), c_void_p(arg572_1.data_ptr()), c_void_p(arg573_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()))
    del arg259_1
    del arg260_1
    del arg572_1
    del arg573_1
    # Source Nodes: [x_334, x_336, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf205 = extern_kernels.convolution(buf204, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf205, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg261_1
    del buf204
    buf206 = buf199; del buf199  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_85(c_void_p(buf206.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg575_1.data_ptr()), c_void_p(arg576_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()))
    del arg262_1
    del arg263_1
    del arg575_1
    del arg576_1
    del buf205
    # Source Nodes: [x_342], Original ATen: [aten.convolution]
    buf207 = extern_kernels.convolution(buf206, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf207, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg264_1
    buf208 = buf207; del buf207  # reuse
    buf209 = buf202; del buf202  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_86(c_void_p(buf208.data_ptr()), c_void_p(arg578_1.data_ptr()), c_void_p(arg579_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg265_1
    del arg266_1
    del arg267_1
    del arg578_1
    del arg579_1
    # Source Nodes: [x_343, x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf210, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf208
    buf211 = buf210; del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_87(c_void_p(buf211.data_ptr()), c_void_p(arg581_1.data_ptr()), c_void_p(arg582_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()))
    del arg268_1
    del arg269_1
    del arg581_1
    del arg582_1
    # Source Nodes: [x_346, x_348, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf212 = extern_kernels.convolution(buf211, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf212, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg270_1
    del buf211
    buf213 = buf206; del buf206  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_88(c_void_p(buf213.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg584_1.data_ptr()), c_void_p(arg585_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg584_1
    del arg585_1
    del buf212
    # Source Nodes: [x_354], Original ATen: [aten.convolution]
    buf214 = extern_kernels.convolution(buf213, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf214, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del arg273_1
    buf215 = buf214; del buf214  # reuse
    buf216 = buf209; del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_89(c_void_p(buf215.data_ptr()), c_void_p(arg587_1.data_ptr()), c_void_p(arg588_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg274_1
    del arg275_1
    del arg276_1
    del arg587_1
    del arg588_1
    # Source Nodes: [x_355, x_356, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf217 = extern_kernels.convolution(buf215, buf216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf217, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    del buf215
    del buf216
    buf218 = buf217; del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_90(c_void_p(buf218.data_ptr()), c_void_p(arg590_1.data_ptr()), c_void_p(arg591_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg590_1
    del arg591_1
    # Source Nodes: [x_358, x_360, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf219 = extern_kernels.convolution(buf218, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf219, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    del arg279_1
    del buf218
    buf220 = buf213; del buf213  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_91(c_void_p(buf220.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg593_1.data_ptr()), c_void_p(arg594_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()))
    del arg280_1
    del arg281_1
    del arg593_1
    del arg594_1
    del buf219
    # Source Nodes: [x_367], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf221, (8, 4096, 14, 14), (802816, 1, 57344, 4096))
    del arg282_1
    buf222 = buf221; del buf221  # reuse
    buf223 = empty_strided((4096, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_92(c_void_p(buf222.data_ptr()), c_void_p(arg596_1.data_ptr()), c_void_p(arg597_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg283_1
    del arg284_1
    del arg285_1
    del arg596_1
    del arg597_1
    # Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf224 = extern_kernels.convolution(buf222, buf223, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf224, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
    del buf222
    buf225 = buf224; del buf224  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_93(c_void_p(buf225.data_ptr()), c_void_p(arg599_1.data_ptr()), c_void_p(arg600_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()))
    del arg286_1
    del arg287_1
    del arg599_1
    del arg600_1
    # Source Nodes: [x_371, x_373, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf226 = extern_kernels.convolution(buf225, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf226, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg288_1
    del buf225
    # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
    buf227 = extern_kernels.convolution(buf220, arg291_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf227, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg291_1
    del buf220
    buf228 = buf226; del buf226  # reuse
    buf229 = buf228; del buf228  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_94(c_void_p(buf229.data_ptr()), c_void_p(arg602_1.data_ptr()), c_void_p(arg603_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(arg605_1.data_ptr()), c_void_p(arg606_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()))
    del arg289_1
    del arg290_1
    del arg292_1
    del arg293_1
    del arg602_1
    del arg603_1
    del arg605_1
    del arg606_1
    del buf227
    # Source Nodes: [shortcut_35, x_379], Original ATen: [aten.convolution, aten.relu]
    buf230 = extern_kernels.convolution(buf229, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf230, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
    del arg294_1
    buf231 = buf230; del buf230  # reuse
    buf232 = buf223; del buf223  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_95(c_void_p(buf231.data_ptr()), c_void_p(arg608_1.data_ptr()), c_void_p(arg609_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg295_1
    del arg296_1
    del arg297_1
    del arg608_1
    del arg609_1
    # Source Nodes: [x_380, x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf233 = extern_kernels.convolution(buf231, buf232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf233, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
    del buf231
    buf234 = buf233; del buf233  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_96(c_void_p(buf234.data_ptr()), c_void_p(arg611_1.data_ptr()), c_void_p(arg612_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()))
    del arg298_1
    del arg299_1
    del arg611_1
    del arg612_1
    # Source Nodes: [x_383, x_385, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf235 = extern_kernels.convolution(buf234, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg300_1
    del buf234
    buf236 = buf229; del buf229  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_97(c_void_p(buf236.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(arg614_1.data_ptr()), c_void_p(arg615_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()))
    del arg301_1
    del arg302_1
    del arg614_1
    del arg615_1
    del buf235
    # Source Nodes: [x_391], Original ATen: [aten.convolution]
    buf237 = extern_kernels.convolution(buf236, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf237, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
    del arg303_1
    buf238 = buf237; del buf237  # reuse
    buf239 = buf232; del buf232  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_98(c_void_p(buf238.data_ptr()), c_void_p(arg617_1.data_ptr()), c_void_p(arg618_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(buf239.data_ptr()))
    del arg304_1
    del arg305_1
    del arg306_1
    del arg617_1
    del arg618_1
    # Source Nodes: [x_392, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf240 = extern_kernels.convolution(buf238, buf239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf240, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
    del buf238
    del buf239
    buf241 = buf240; del buf240  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_99(c_void_p(buf241.data_ptr()), c_void_p(arg620_1.data_ptr()), c_void_p(arg621_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()))
    del arg307_1
    del arg308_1
    del arg620_1
    del arg621_1
    # Source Nodes: [x_395, x_397, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf242 = extern_kernels.convolution(buf241, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    del arg309_1
    del buf241
    buf243 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf244 = reinterpret_tensor(buf243, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf243  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_100(c_void_p(buf244.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg623_1.data_ptr()), c_void_p(arg624_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(buf236.data_ptr()))
    del arg310_1
    del arg311_1
    del arg623_1
    del arg624_1
    del buf236
    del buf242
    buf245 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_408], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg313_1, reinterpret_tensor(buf244, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg312_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf245)
    del arg312_1
    del arg313_1
    return (buf245, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg317_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg320_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg323_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg326_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg329_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg332_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg335_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg338_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg341_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg344_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg347_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg350_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg353_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg356_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg359_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg362_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg365_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg368_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg371_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg374_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg377_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg380_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg383_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg386_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg389_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg392_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg395_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg398_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg401_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg404_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg407_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg410_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg413_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg416_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg419_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg422_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg425_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg428_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg431_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg434_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg437_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg440_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg443_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg446_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg449_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg452_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg455_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg458_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg461_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg464_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg467_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg470_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg473_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg476_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg479_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg482_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg485_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg488_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg491_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg494_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg497_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg500_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg503_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg506_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg509_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg512_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg515_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg518_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg521_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg524_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg527_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg530_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg533_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg536_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg539_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg542_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg545_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg548_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg551_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg554_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg557_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg558_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg560_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg561_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg563_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg564_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg566_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg567_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg569_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg570_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg572_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg573_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg575_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg576_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg578_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg579_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg581_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg582_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg584_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg585_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg587_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg588_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg590_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg591_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg593_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg594_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg596_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg597_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg599_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg600_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg602_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg603_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg605_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg606_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg608_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg609_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg611_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg612_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg614_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg615_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg617_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg618_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg620_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg621_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg623_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg624_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg626_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swsl_resnext101_32x16d', benchmark_compiled_module)
