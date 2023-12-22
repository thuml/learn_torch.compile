
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x2) + (75264L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (24L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x2) + (43904L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (56L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(351232L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(238336L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_51 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_56 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)), static_cast<long>(8L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (72L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (8L*x2) + (72L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (24, ), (1, ))
    assert_size_stride(arg3_1, (24, ), (1, ))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (24, ), (1, ))
    assert_size_stride(arg6_1, (24, ), (1, ))
    assert_size_stride(arg7_1, (24, ), (1, ))
    assert_size_stride(arg8_1, (24, ), (1, ))
    assert_size_stride(arg9_1, (24, ), (1, ))
    assert_size_stride(arg10_1, (56, ), (1, ))
    assert_size_stride(arg11_1, (56, ), (1, ))
    assert_size_stride(arg12_1, (56, ), (1, ))
    assert_size_stride(arg13_1, (56, ), (1, ))
    assert_size_stride(arg14_1, (56, ), (1, ))
    assert_size_stride(arg15_1, (56, ), (1, ))
    assert_size_stride(arg16_1, (56, ), (1, ))
    assert_size_stride(arg17_1, (56, ), (1, ))
    assert_size_stride(arg18_1, (152, ), (1, ))
    assert_size_stride(arg19_1, (152, ), (1, ))
    assert_size_stride(arg20_1, (152, ), (1, ))
    assert_size_stride(arg21_1, (152, ), (1, ))
    assert_size_stride(arg22_1, (152, ), (1, ))
    assert_size_stride(arg23_1, (152, ), (1, ))
    assert_size_stride(arg24_1, (152, ), (1, ))
    assert_size_stride(arg25_1, (152, ), (1, ))
    assert_size_stride(arg26_1, (152, ), (1, ))
    assert_size_stride(arg27_1, (152, ), (1, ))
    assert_size_stride(arg28_1, (152, ), (1, ))
    assert_size_stride(arg29_1, (152, ), (1, ))
    assert_size_stride(arg30_1, (152, ), (1, ))
    assert_size_stride(arg31_1, (152, ), (1, ))
    assert_size_stride(arg32_1, (152, ), (1, ))
    assert_size_stride(arg33_1, (152, ), (1, ))
    assert_size_stride(arg34_1, (152, ), (1, ))
    assert_size_stride(arg35_1, (152, ), (1, ))
    assert_size_stride(arg36_1, (152, ), (1, ))
    assert_size_stride(arg37_1, (152, ), (1, ))
    assert_size_stride(arg38_1, (152, ), (1, ))
    assert_size_stride(arg39_1, (152, ), (1, ))
    assert_size_stride(arg40_1, (152, ), (1, ))
    assert_size_stride(arg41_1, (152, ), (1, ))
    assert_size_stride(arg42_1, (152, ), (1, ))
    assert_size_stride(arg43_1, (152, ), (1, ))
    assert_size_stride(arg44_1, (368, ), (1, ))
    assert_size_stride(arg45_1, (368, ), (1, ))
    assert_size_stride(arg46_1, (368, ), (1, ))
    assert_size_stride(arg47_1, (368, ), (1, ))
    assert_size_stride(arg48_1, (368, ), (1, ))
    assert_size_stride(arg49_1, (368, ), (1, ))
    assert_size_stride(arg50_1, (368, ), (1, ))
    assert_size_stride(arg51_1, (368, ), (1, ))
    assert_size_stride(arg52_1, (368, ), (1, ))
    assert_size_stride(arg53_1, (368, ), (1, ))
    assert_size_stride(arg54_1, (368, ), (1, ))
    assert_size_stride(arg55_1, (368, ), (1, ))
    assert_size_stride(arg56_1, (368, ), (1, ))
    assert_size_stride(arg57_1, (368, ), (1, ))
    assert_size_stride(arg58_1, (368, ), (1, ))
    assert_size_stride(arg59_1, (368, ), (1, ))
    assert_size_stride(arg60_1, (368, ), (1, ))
    assert_size_stride(arg61_1, (368, ), (1, ))
    assert_size_stride(arg62_1, (368, ), (1, ))
    assert_size_stride(arg63_1, (368, ), (1, ))
    assert_size_stride(arg64_1, (368, ), (1, ))
    assert_size_stride(arg65_1, (368, ), (1, ))
    assert_size_stride(arg66_1, (368, ), (1, ))
    assert_size_stride(arg67_1, (368, ), (1, ))
    assert_size_stride(arg68_1, (368, ), (1, ))
    assert_size_stride(arg69_1, (368, ), (1, ))
    assert_size_stride(arg70_1, (368, ), (1, ))
    assert_size_stride(arg71_1, (368, ), (1, ))
    assert_size_stride(arg72_1, (368, ), (1, ))
    assert_size_stride(arg73_1, (368, ), (1, ))
    assert_size_stride(arg74_1, (368, ), (1, ))
    assert_size_stride(arg75_1, (368, ), (1, ))
    assert_size_stride(arg76_1, (368, ), (1, ))
    assert_size_stride(arg77_1, (368, ), (1, ))
    assert_size_stride(arg78_1, (368, ), (1, ))
    assert_size_stride(arg79_1, (368, ), (1, ))
    assert_size_stride(arg80_1, (368, ), (1, ))
    assert_size_stride(arg81_1, (368, ), (1, ))
    assert_size_stride(arg82_1, (368, ), (1, ))
    assert_size_stride(arg83_1, (368, ), (1, ))
    assert_size_stride(arg84_1, (368, ), (1, ))
    assert_size_stride(arg85_1, (368, ), (1, ))
    assert_size_stride(arg86_1, (368, ), (1, ))
    assert_size_stride(arg87_1, (368, ), (1, ))
    assert_size_stride(arg88_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg89_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg90_1, (24, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg91_1, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg92_1, (8, ), (1, ))
    assert_size_stride(arg93_1, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg94_1, (24, ), (1, ))
    assert_size_stride(arg95_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg96_1, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg97_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg98_1, (56, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg99_1, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg100_1, (6, ), (1, ))
    assert_size_stride(arg101_1, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg102_1, (56, ), (1, ))
    assert_size_stride(arg103_1, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg104_1, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg105_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg106_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg107_1, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg108_1, (14, ), (1, ))
    assert_size_stride(arg109_1, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg110_1, (152, ), (1, ))
    assert_size_stride(arg111_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg112_1, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg113_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg114_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg115_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg116_1, (38, ), (1, ))
    assert_size_stride(arg117_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg118_1, (152, ), (1, ))
    assert_size_stride(arg119_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg120_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg121_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg122_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg123_1, (38, ), (1, ))
    assert_size_stride(arg124_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg125_1, (152, ), (1, ))
    assert_size_stride(arg126_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg127_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg128_1, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg129_1, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg130_1, (38, ), (1, ))
    assert_size_stride(arg131_1, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg132_1, (152, ), (1, ))
    assert_size_stride(arg133_1, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg134_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg135_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg136_1, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg137_1, (38, ), (1, ))
    assert_size_stride(arg138_1, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg139_1, (368, ), (1, ))
    assert_size_stride(arg140_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg141_1, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg142_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg143_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg144_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg145_1, (92, ), (1, ))
    assert_size_stride(arg146_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg147_1, (368, ), (1, ))
    assert_size_stride(arg148_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg149_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg150_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg151_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg152_1, (92, ), (1, ))
    assert_size_stride(arg153_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg154_1, (368, ), (1, ))
    assert_size_stride(arg155_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg156_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg157_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg158_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg159_1, (92, ), (1, ))
    assert_size_stride(arg160_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg161_1, (368, ), (1, ))
    assert_size_stride(arg162_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg163_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg164_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg165_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg166_1, (92, ), (1, ))
    assert_size_stride(arg167_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg168_1, (368, ), (1, ))
    assert_size_stride(arg169_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg170_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg171_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg172_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg173_1, (92, ), (1, ))
    assert_size_stride(arg174_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg175_1, (368, ), (1, ))
    assert_size_stride(arg176_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg177_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg178_1, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg179_1, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg180_1, (92, ), (1, ))
    assert_size_stride(arg181_1, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(arg182_1, (368, ), (1, ))
    assert_size_stride(arg183_1, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(arg184_1, (1000, 368), (368, 1))
    assert_size_stride(arg185_1, (1000, ), (1, ))
    assert_size_stride(arg186_1, (32, ), (1, ))
    assert_size_stride(arg187_1, (32, ), (1, ))
    assert_size_stride(arg188_1, (24, ), (1, ))
    assert_size_stride(arg189_1, (24, ), (1, ))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (24, ), (1, ))
    assert_size_stride(arg194_1, (24, ), (1, ))
    assert_size_stride(arg195_1, (24, ), (1, ))
    assert_size_stride(arg196_1, (56, ), (1, ))
    assert_size_stride(arg197_1, (56, ), (1, ))
    assert_size_stride(arg198_1, (56, ), (1, ))
    assert_size_stride(arg199_1, (56, ), (1, ))
    assert_size_stride(arg200_1, (56, ), (1, ))
    assert_size_stride(arg201_1, (56, ), (1, ))
    assert_size_stride(arg202_1, (56, ), (1, ))
    assert_size_stride(arg203_1, (56, ), (1, ))
    assert_size_stride(arg204_1, (152, ), (1, ))
    assert_size_stride(arg205_1, (152, ), (1, ))
    assert_size_stride(arg206_1, (152, ), (1, ))
    assert_size_stride(arg207_1, (152, ), (1, ))
    assert_size_stride(arg208_1, (152, ), (1, ))
    assert_size_stride(arg209_1, (152, ), (1, ))
    assert_size_stride(arg210_1, (152, ), (1, ))
    assert_size_stride(arg211_1, (152, ), (1, ))
    assert_size_stride(arg212_1, (152, ), (1, ))
    assert_size_stride(arg213_1, (152, ), (1, ))
    assert_size_stride(arg214_1, (152, ), (1, ))
    assert_size_stride(arg215_1, (152, ), (1, ))
    assert_size_stride(arg216_1, (152, ), (1, ))
    assert_size_stride(arg217_1, (152, ), (1, ))
    assert_size_stride(arg218_1, (152, ), (1, ))
    assert_size_stride(arg219_1, (152, ), (1, ))
    assert_size_stride(arg220_1, (152, ), (1, ))
    assert_size_stride(arg221_1, (152, ), (1, ))
    assert_size_stride(arg222_1, (152, ), (1, ))
    assert_size_stride(arg223_1, (152, ), (1, ))
    assert_size_stride(arg224_1, (152, ), (1, ))
    assert_size_stride(arg225_1, (152, ), (1, ))
    assert_size_stride(arg226_1, (152, ), (1, ))
    assert_size_stride(arg227_1, (152, ), (1, ))
    assert_size_stride(arg228_1, (152, ), (1, ))
    assert_size_stride(arg229_1, (152, ), (1, ))
    assert_size_stride(arg230_1, (368, ), (1, ))
    assert_size_stride(arg231_1, (368, ), (1, ))
    assert_size_stride(arg232_1, (368, ), (1, ))
    assert_size_stride(arg233_1, (368, ), (1, ))
    assert_size_stride(arg234_1, (368, ), (1, ))
    assert_size_stride(arg235_1, (368, ), (1, ))
    assert_size_stride(arg236_1, (368, ), (1, ))
    assert_size_stride(arg237_1, (368, ), (1, ))
    assert_size_stride(arg238_1, (368, ), (1, ))
    assert_size_stride(arg239_1, (368, ), (1, ))
    assert_size_stride(arg240_1, (368, ), (1, ))
    assert_size_stride(arg241_1, (368, ), (1, ))
    assert_size_stride(arg242_1, (368, ), (1, ))
    assert_size_stride(arg243_1, (368, ), (1, ))
    assert_size_stride(arg244_1, (368, ), (1, ))
    assert_size_stride(arg245_1, (368, ), (1, ))
    assert_size_stride(arg246_1, (368, ), (1, ))
    assert_size_stride(arg247_1, (368, ), (1, ))
    assert_size_stride(arg248_1, (368, ), (1, ))
    assert_size_stride(arg249_1, (368, ), (1, ))
    assert_size_stride(arg250_1, (368, ), (1, ))
    assert_size_stride(arg251_1, (368, ), (1, ))
    assert_size_stride(arg252_1, (368, ), (1, ))
    assert_size_stride(arg253_1, (368, ), (1, ))
    assert_size_stride(arg254_1, (368, ), (1, ))
    assert_size_stride(arg255_1, (368, ), (1, ))
    assert_size_stride(arg256_1, (368, ), (1, ))
    assert_size_stride(arg257_1, (368, ), (1, ))
    assert_size_stride(arg258_1, (368, ), (1, ))
    assert_size_stride(arg259_1, (368, ), (1, ))
    assert_size_stride(arg260_1, (368, ), (1, ))
    assert_size_stride(arg261_1, (368, ), (1, ))
    assert_size_stride(arg262_1, (368, ), (1, ))
    assert_size_stride(arg263_1, (368, ), (1, ))
    assert_size_stride(arg264_1, (368, ), (1, ))
    assert_size_stride(arg265_1, (368, ), (1, ))
    assert_size_stride(arg266_1, (368, ), (1, ))
    assert_size_stride(arg267_1, (368, ), (1, ))
    assert_size_stride(arg268_1, (368, ), (1, ))
    assert_size_stride(arg269_1, (368, ), (1, ))
    assert_size_stride(arg270_1, (368, ), (1, ))
    assert_size_stride(arg271_1, (368, ), (1, ))
    assert_size_stride(arg272_1, (368, ), (1, ))
    assert_size_stride(arg273_1, (368, ), (1, ))
    assert_size_stride(arg274_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg274_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg274_1
    del arg88_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg186_1
    del arg187_1
    del arg1_1
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf4, (8, 24, 112, 112), (301056, 1, 2688, 24))
    del arg89_1
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((24, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg188_1
    del arg189_1
    del arg2_1
    del arg3_1
    del arg90_1
    # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf7 = extern_kernels.convolution(buf5, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
    assert_size_stride(buf7, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del buf5
    del buf6
    buf8 = buf7; del buf7  # reuse
    buf9 = empty_strided((8, 24, 1, 1), (24, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf9, (8, 24, 1, 1), (24, 1, 24, 24), 0); del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_3(c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg190_1
    del arg191_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
    buf11 = extern_kernels.convolution(buf10, arg91_1, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (8, 8, 1, 1), (8, 1, 8, 8))
    del arg91_1
    del arg92_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    cpp_fused_relu_4(c_void_p(buf12.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.relu]
    buf13 = extern_kernels.convolution(buf12, arg93_1, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf13, (8, 24, 1, 1), (24, 1, 24, 24))
    del arg93_1
    del arg94_1
    del buf12
    buf14 = buf8; del buf8  # reuse
    cpp_fused_mul_sigmoid_5(c_void_p(buf14.data_ptr()), c_void_p(buf13.data_ptr()))
    del buf13
    # Source Nodes: [sigmoid, x_18, x_19], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf15 = extern_kernels.convolution(buf14, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg95_1
    del buf14
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf3, arg96_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg96_1
    del buf3
    buf17 = buf15; del buf15  # reuse
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_6(c_void_p(buf18.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg194_1
    del arg195_1
    del arg6_1
    del arg7_1
    del arg8_1
    del arg9_1
    del buf16
    # Source Nodes: [shortcut_1, x_34], Original ATen: [aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf18, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 56, 56, 56), (175616, 1, 3136, 56))
    del arg97_1
    buf20 = buf19; del buf19  # reuse
    buf21 = empty_strided((56, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7(c_void_p(buf20.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg10_1
    del arg11_1
    del arg196_1
    del arg197_1
    del arg98_1
    # Source Nodes: [x_35, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf22 = extern_kernels.convolution(buf20, buf21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=7, bias=None)
    assert_size_stride(buf22, (8, 56, 28, 28), (43904, 1, 1568, 56))
    del buf20
    del buf21
    buf23 = buf22; del buf22  # reuse
    buf24 = empty_strided((8, 56, 1, 1), (56, 1, 448, 448), device='cpu', dtype=torch.float32)
    buf25 = reinterpret_tensor(buf24, (8, 56, 1, 1), (56, 1, 56, 56), 0); del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_8(c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg198_1
    del arg199_1
    # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
    buf26 = extern_kernels.convolution(buf25, arg99_1, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (8, 6, 1, 1), (6, 1, 6, 6))
    del arg100_1
    del arg99_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    cpp_fused_relu_9(c_void_p(buf27.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.relu]
    buf28 = extern_kernels.convolution(buf27, arg101_1, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf28, (8, 56, 1, 1), (56, 1, 56, 56))
    del arg101_1
    del arg102_1
    del buf27
    buf29 = buf23; del buf23  # reuse
    cpp_fused_mul_sigmoid_10(c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()))
    del buf28
    # Source Nodes: [sigmoid_1, x_46, x_47], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf30 = extern_kernels.convolution(buf29, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (8, 56, 28, 28), (43904, 1, 1568, 56))
    del arg103_1
    del buf29
    # Source Nodes: [x_53], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf18, arg104_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 56, 28, 28), (43904, 1, 1568, 56))
    del arg104_1
    del buf18
    buf32 = buf30; del buf30  # reuse
    buf33 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_11(c_void_p(buf33.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg16_1
    del arg17_1
    del arg200_1
    del arg201_1
    del arg202_1
    del arg203_1
    del buf31
    # Source Nodes: [shortcut_2, x_62], Original ATen: [aten.convolution, aten.relu]
    buf34 = extern_kernels.convolution(buf33, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (8, 152, 28, 28), (119168, 1, 4256, 152))
    del arg105_1
    buf35 = buf34; del buf34  # reuse
    buf36 = empty_strided((152, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12(c_void_p(buf35.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg106_1
    del arg18_1
    del arg19_1
    del arg204_1
    del arg205_1
    # Source Nodes: [x_63, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf37 = extern_kernels.convolution(buf35, buf36, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
    assert_size_stride(buf37, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del buf35
    buf38 = buf37; del buf37  # reuse
    buf39 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cpu', dtype=torch.float32)
    buf40 = reinterpret_tensor(buf39, (8, 152, 1, 1), (152, 1, 152, 152), 0); del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_13(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg206_1
    del arg207_1
    del arg20_1
    del arg21_1
    # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
    buf41 = extern_kernels.convolution(buf40, arg107_1, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (8, 14, 1, 1), (14, 1, 14, 14))
    del arg107_1
    del arg108_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    cpp_fused_relu_14(c_void_p(buf42.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.relu]
    buf43 = extern_kernels.convolution(buf42, arg109_1, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf43, (8, 152, 1, 1), (152, 1, 152, 152))
    del arg109_1
    del arg110_1
    del buf42
    buf44 = buf38; del buf38  # reuse
    cpp_fused_mul_sigmoid_15(c_void_p(buf44.data_ptr()), c_void_p(buf43.data_ptr()))
    # Source Nodes: [sigmoid_2, x_74, x_75], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf45 = extern_kernels.convolution(buf44, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg111_1
    del buf44
    # Source Nodes: [x_81], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf33, arg112_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg112_1
    del buf33
    buf47 = buf45; del buf45  # reuse
    buf48 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf48.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg208_1
    del arg209_1
    del arg210_1
    del arg211_1
    del arg22_1
    del arg23_1
    del arg24_1
    del arg25_1
    del buf46
    # Source Nodes: [shortcut_3, x_89], Original ATen: [aten.convolution, aten.relu]
    buf49 = extern_kernels.convolution(buf48, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg113_1
    buf50 = buf49; del buf49  # reuse
    buf51 = buf36; del buf36  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf50.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg114_1
    del arg212_1
    del arg213_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
    assert_size_stride(buf52, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del buf50
    buf53 = buf52; del buf52  # reuse
    buf54 = reinterpret_tensor(buf43, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf43  # reuse
    buf55 = reinterpret_tensor(buf54, (8, 152, 1, 1), (152, 1, 152, 152), 0); del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_18(c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg28_1
    del arg29_1
    # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
    buf56 = extern_kernels.convolution(buf55, arg115_1, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf56, (8, 38, 1, 1), (38, 1, 38, 38))
    del arg115_1
    del arg116_1
    del buf55
    buf57 = buf56; del buf56  # reuse
    cpp_fused_relu_19(c_void_p(buf57.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf57, arg117_1, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (8, 152, 1, 1), (152, 1, 152, 152))
    del arg117_1
    del arg118_1
    del buf57
    buf59 = buf53; del buf53  # reuse
    cpp_fused_mul_sigmoid_20(c_void_p(buf59.data_ptr()), c_void_p(buf58.data_ptr()))
    # Source Nodes: [sigmoid_3, x_101, x_102], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf60 = extern_kernels.convolution(buf59, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg119_1
    del buf59
    buf61 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_21(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg216_1
    del arg217_1
    del arg30_1
    del arg31_1
    del buf60
    # Source Nodes: [x_111], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg120_1
    buf63 = buf62; del buf62  # reuse
    buf64 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22(c_void_p(buf63.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg121_1
    del arg218_1
    del arg219_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_112, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf65 = extern_kernels.convolution(buf63, buf64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
    assert_size_stride(buf65, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del buf63
    buf66 = buf65; del buf65  # reuse
    buf67 = reinterpret_tensor(buf58, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf58  # reuse
    buf68 = reinterpret_tensor(buf67, (8, 152, 1, 1), (152, 1, 152, 152), 0); del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_23(c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg220_1
    del arg221_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
    buf69 = extern_kernels.convolution(buf68, arg122_1, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf69, (8, 38, 1, 1), (38, 1, 38, 38))
    del arg122_1
    del arg123_1
    del buf68
    buf70 = buf69; del buf69  # reuse
    cpp_fused_relu_24(c_void_p(buf70.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.relu]
    buf71 = extern_kernels.convolution(buf70, arg124_1, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf71, (8, 152, 1, 1), (152, 1, 152, 152))
    del arg124_1
    del arg125_1
    del buf70
    buf72 = buf66; del buf66  # reuse
    cpp_fused_mul_sigmoid_25(c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()))
    # Source Nodes: [sigmoid_4, x_123, x_124], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf73 = extern_kernels.convolution(buf72, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg126_1
    del buf72
    buf74 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_26(c_void_p(buf74.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg222_1
    del arg223_1
    del arg36_1
    del arg37_1
    del buf73
    # Source Nodes: [x_133], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg127_1
    buf76 = buf75; del buf75  # reuse
    buf77 = buf64; del buf64  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27(c_void_p(buf76.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg128_1
    del arg224_1
    del arg225_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_134, x_138, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
    assert_size_stride(buf78, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del buf76
    del buf77
    buf79 = buf78; del buf78  # reuse
    buf80 = reinterpret_tensor(buf71, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf71  # reuse
    buf81 = reinterpret_tensor(buf80, (8, 152, 1, 1), (152, 1, 152, 152), 0); del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_28(c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
    buf82 = extern_kernels.convolution(buf81, arg129_1, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (8, 38, 1, 1), (38, 1, 38, 38))
    del arg129_1
    del arg130_1
    del buf81
    buf83 = buf82; del buf82  # reuse
    cpp_fused_relu_29(c_void_p(buf83.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.relu]
    buf84 = extern_kernels.convolution(buf83, arg131_1, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf84, (8, 152, 1, 1), (152, 1, 152, 152))
    del arg131_1
    del arg132_1
    del buf83
    buf85 = buf79; del buf79  # reuse
    cpp_fused_mul_sigmoid_30(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()))
    del buf84
    # Source Nodes: [sigmoid_5, x_145, x_146], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf86 = extern_kernels.convolution(buf85, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 152, 14, 14), (29792, 1, 2128, 152))
    del arg133_1
    del buf85
    buf87 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_31(c_void_p(buf87.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg228_1
    del arg229_1
    del arg42_1
    del arg43_1
    del buf86
    # Source Nodes: [x_156], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 368, 14, 14), (72128, 1, 5152, 368))
    del arg134_1
    buf89 = buf88; del buf88  # reuse
    buf90 = empty_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf89.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg135_1
    del arg230_1
    del arg231_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_157, x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf91 = extern_kernels.convolution(buf89, buf90, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf91, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf89
    buf92 = buf91; del buf91  # reuse
    buf93 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cpu', dtype=torch.float32)
    buf94 = reinterpret_tensor(buf93, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_33(c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg232_1
    del arg233_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
    buf95 = extern_kernels.convolution(buf94, arg136_1, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf95, (8, 38, 1, 1), (38, 1, 38, 38))
    del arg136_1
    del arg137_1
    del buf94
    buf96 = buf95; del buf95  # reuse
    cpp_fused_relu_34(c_void_p(buf96.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.relu]
    buf97 = extern_kernels.convolution(buf96, arg138_1, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf97, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg138_1
    del arg139_1
    del buf96
    buf98 = buf92; del buf92  # reuse
    cpp_fused_mul_sigmoid_35(c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()))
    # Source Nodes: [sigmoid_6, x_168, x_169], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf99 = extern_kernels.convolution(buf98, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg140_1
    del buf98
    # Source Nodes: [x_175], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf87, arg141_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg141_1
    del buf87
    buf101 = buf100; del buf100  # reuse
    buf102 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_36(c_void_p(buf102.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg234_1
    del arg235_1
    del arg236_1
    del arg237_1
    del arg48_1
    del arg49_1
    del arg50_1
    del arg51_1
    del buf99
    # Source Nodes: [shortcut_7, x_183], Original ATen: [aten.convolution, aten.relu]
    buf103 = extern_kernels.convolution(buf102, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg142_1
    buf104 = buf103; del buf103  # reuse
    buf105 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37(c_void_p(buf104.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg143_1
    del arg238_1
    del arg239_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_184, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf106, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf104
    buf107 = buf106; del buf106  # reuse
    buf108 = reinterpret_tensor(buf97, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf97  # reuse
    buf109 = reinterpret_tensor(buf108, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_38(c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg240_1
    del arg241_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
    buf110 = extern_kernels.convolution(buf109, arg144_1, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf110, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg144_1
    del arg145_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    cpp_fused_relu_39(c_void_p(buf111.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf111, arg146_1, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf112, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg146_1
    del arg147_1
    del buf111
    buf113 = buf107; del buf107  # reuse
    cpp_fused_mul_sigmoid_40(c_void_p(buf113.data_ptr()), c_void_p(buf112.data_ptr()))
    # Source Nodes: [sigmoid_7, x_195, x_196], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf114 = extern_kernels.convolution(buf113, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg148_1
    del buf113
    buf115 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_41(c_void_p(buf115.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg56_1
    del arg57_1
    del buf114
    # Source Nodes: [x_205], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg149_1
    buf117 = buf116; del buf116  # reuse
    buf118 = buf105; del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42(c_void_p(buf117.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg150_1
    del arg244_1
    del arg245_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_206, x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf119, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf117
    buf120 = buf119; del buf119  # reuse
    buf121 = reinterpret_tensor(buf112, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf112  # reuse
    buf122 = reinterpret_tensor(buf121, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf121  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_43(c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg246_1
    del arg247_1
    del arg60_1
    del arg61_1
    # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
    buf123 = extern_kernels.convolution(buf122, arg151_1, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf123, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg151_1
    del arg152_1
    del buf122
    buf124 = buf123; del buf123  # reuse
    cpp_fused_relu_44(c_void_p(buf124.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.relu]
    buf125 = extern_kernels.convolution(buf124, arg153_1, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf125, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg153_1
    del arg154_1
    del buf124
    buf126 = buf120; del buf120  # reuse
    cpp_fused_mul_sigmoid_45(c_void_p(buf126.data_ptr()), c_void_p(buf125.data_ptr()))
    # Source Nodes: [sigmoid_8, x_217, x_218], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf127 = extern_kernels.convolution(buf126, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg155_1
    del buf126
    buf128 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf128.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg248_1
    del arg249_1
    del arg62_1
    del arg63_1
    del buf127
    # Source Nodes: [x_227], Original ATen: [aten.convolution]
    buf129 = extern_kernels.convolution(buf128, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf129, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg156_1
    buf130 = buf129; del buf129  # reuse
    buf131 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47(c_void_p(buf130.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg157_1
    del arg250_1
    del arg251_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_228, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf132 = extern_kernels.convolution(buf130, buf131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf132, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf130
    buf133 = buf132; del buf132  # reuse
    buf134 = reinterpret_tensor(buf125, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf125  # reuse
    buf135 = reinterpret_tensor(buf134, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf134  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_48(c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg252_1
    del arg253_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
    buf136 = extern_kernels.convolution(buf135, arg158_1, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf136, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg158_1
    del arg159_1
    del buf135
    buf137 = buf136; del buf136  # reuse
    cpp_fused_relu_49(c_void_p(buf137.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.relu]
    buf138 = extern_kernels.convolution(buf137, arg160_1, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf138, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg160_1
    del arg161_1
    del buf137
    buf139 = buf133; del buf133  # reuse
    cpp_fused_mul_sigmoid_50(c_void_p(buf139.data_ptr()), c_void_p(buf138.data_ptr()))
    # Source Nodes: [sigmoid_9, x_239, x_240], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf140 = extern_kernels.convolution(buf139, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg162_1
    del buf139
    buf141 = buf128; del buf128  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_51(c_void_p(buf141.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg254_1
    del arg255_1
    del arg68_1
    del arg69_1
    del buf140
    # Source Nodes: [x_249], Original ATen: [aten.convolution]
    buf142 = extern_kernels.convolution(buf141, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg163_1
    buf143 = buf142; del buf142  # reuse
    buf144 = buf131; del buf131  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52(c_void_p(buf143.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg164_1
    del arg256_1
    del arg257_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_250, x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf145 = extern_kernels.convolution(buf143, buf144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf145, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf143
    buf146 = buf145; del buf145  # reuse
    buf147 = reinterpret_tensor(buf138, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf138  # reuse
    buf148 = reinterpret_tensor(buf147, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf147  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_53(c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg258_1
    del arg259_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
    buf149 = extern_kernels.convolution(buf148, arg165_1, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf149, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg165_1
    del arg166_1
    del buf148
    buf150 = buf149; del buf149  # reuse
    cpp_fused_relu_54(c_void_p(buf150.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.relu]
    buf151 = extern_kernels.convolution(buf150, arg167_1, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf151, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg167_1
    del arg168_1
    del buf150
    buf152 = buf146; del buf146  # reuse
    cpp_fused_mul_sigmoid_55(c_void_p(buf152.data_ptr()), c_void_p(buf151.data_ptr()))
    # Source Nodes: [sigmoid_10, x_261, x_262], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf153 = extern_kernels.convolution(buf152, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf153, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg169_1
    del buf152
    buf154 = buf141; del buf141  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_56(c_void_p(buf154.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg260_1
    del arg261_1
    del arg74_1
    del arg75_1
    del buf153
    # Source Nodes: [x_271], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf155, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg170_1
    buf156 = buf155; del buf155  # reuse
    buf157 = buf144; del buf144  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57(c_void_p(buf156.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf157.data_ptr()))
    del arg171_1
    del arg262_1
    del arg263_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_272, x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf158 = extern_kernels.convolution(buf156, buf157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf158, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf156
    buf159 = buf158; del buf158  # reuse
    buf160 = reinterpret_tensor(buf151, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf151  # reuse
    buf161 = reinterpret_tensor(buf160, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf160  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_58(c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg264_1
    del arg265_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
    buf162 = extern_kernels.convolution(buf161, arg172_1, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf162, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg172_1
    del arg173_1
    del buf161
    buf163 = buf162; del buf162  # reuse
    cpp_fused_relu_59(c_void_p(buf163.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.relu]
    buf164 = extern_kernels.convolution(buf163, arg174_1, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf164, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg174_1
    del arg175_1
    del buf163
    buf165 = buf159; del buf159  # reuse
    cpp_fused_mul_sigmoid_60(c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()))
    # Source Nodes: [sigmoid_11, x_283, x_284], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf166 = extern_kernels.convolution(buf165, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf166, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg176_1
    del buf165
    buf167 = buf154; del buf154  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_61(c_void_p(buf167.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg266_1
    del arg267_1
    del arg80_1
    del arg81_1
    del buf166
    # Source Nodes: [x_293], Original ATen: [aten.convolution]
    buf168 = extern_kernels.convolution(buf167, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf168, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg177_1
    buf169 = buf168; del buf168  # reuse
    buf170 = buf157; del buf157  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf169.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf170.data_ptr()))
    del arg178_1
    del arg268_1
    del arg269_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_294, x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
    assert_size_stride(buf171, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del buf169
    del buf170
    buf172 = buf171; del buf171  # reuse
    buf173 = reinterpret_tensor(buf164, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf164  # reuse
    buf174 = reinterpret_tensor(buf173, (8, 368, 1, 1), (368, 1, 368, 368), 0); del buf173  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_63(c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg270_1
    del arg271_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
    buf175 = extern_kernels.convolution(buf174, arg179_1, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf175, (8, 92, 1, 1), (92, 1, 92, 92))
    del arg179_1
    del arg180_1
    del buf174
    buf176 = buf175; del buf175  # reuse
    cpp_fused_relu_64(c_void_p(buf176.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.relu]
    buf177 = extern_kernels.convolution(buf176, arg181_1, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf177, (8, 368, 1, 1), (368, 1, 368, 368))
    del arg181_1
    del arg182_1
    del buf176
    buf178 = buf172; del buf172  # reuse
    cpp_fused_mul_sigmoid_65(c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()))
    # Source Nodes: [sigmoid_12, x_305, x_306], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf179 = extern_kernels.convolution(buf178, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf179, (8, 368, 7, 7), (18032, 1, 2576, 368))
    del arg183_1
    del buf178
    buf180 = reinterpret_tensor(buf177, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf177  # reuse
    buf181 = reinterpret_tensor(buf180, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf180  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_66(c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg272_1
    del arg273_1
    del arg86_1
    del arg87_1
    del buf167
    del buf179
    buf182 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_322], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf181, (8, 368), (368, 1), 0), reinterpret_tensor(arg184_1, (368, 1000), (1, 368), 0), alpha=1, beta=1, out=buf182)
    del arg184_1
    del arg185_1
    return (buf182, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((24, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((56, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((14, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1000, 368), (368, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
