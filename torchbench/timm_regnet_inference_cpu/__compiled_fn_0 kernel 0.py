
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2809856L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
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
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1404928L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
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
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)), static_cast<long>(112L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1008L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (112L*x2) + (1008L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x2) + (109760L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8960L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2240L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(in_out_ptr0 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_96 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2240L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x2) + (109760L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8960L); x0+=static_cast<long>(8L))
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (224, ), (1, ))
    assert_size_stride(arg3_1, (224, ), (1, ))
    assert_size_stride(arg4_1, (224, ), (1, ))
    assert_size_stride(arg5_1, (224, ), (1, ))
    assert_size_stride(arg6_1, (224, ), (1, ))
    assert_size_stride(arg7_1, (224, ), (1, ))
    assert_size_stride(arg8_1, (224, ), (1, ))
    assert_size_stride(arg9_1, (224, ), (1, ))
    assert_size_stride(arg10_1, (224, ), (1, ))
    assert_size_stride(arg11_1, (224, ), (1, ))
    assert_size_stride(arg12_1, (224, ), (1, ))
    assert_size_stride(arg13_1, (224, ), (1, ))
    assert_size_stride(arg14_1, (224, ), (1, ))
    assert_size_stride(arg15_1, (224, ), (1, ))
    assert_size_stride(arg16_1, (448, ), (1, ))
    assert_size_stride(arg17_1, (448, ), (1, ))
    assert_size_stride(arg18_1, (448, ), (1, ))
    assert_size_stride(arg19_1, (448, ), (1, ))
    assert_size_stride(arg20_1, (448, ), (1, ))
    assert_size_stride(arg21_1, (448, ), (1, ))
    assert_size_stride(arg22_1, (448, ), (1, ))
    assert_size_stride(arg23_1, (448, ), (1, ))
    assert_size_stride(arg24_1, (448, ), (1, ))
    assert_size_stride(arg25_1, (448, ), (1, ))
    assert_size_stride(arg26_1, (448, ), (1, ))
    assert_size_stride(arg27_1, (448, ), (1, ))
    assert_size_stride(arg28_1, (448, ), (1, ))
    assert_size_stride(arg29_1, (448, ), (1, ))
    assert_size_stride(arg30_1, (448, ), (1, ))
    assert_size_stride(arg31_1, (448, ), (1, ))
    assert_size_stride(arg32_1, (448, ), (1, ))
    assert_size_stride(arg33_1, (448, ), (1, ))
    assert_size_stride(arg34_1, (448, ), (1, ))
    assert_size_stride(arg35_1, (448, ), (1, ))
    assert_size_stride(arg36_1, (448, ), (1, ))
    assert_size_stride(arg37_1, (448, ), (1, ))
    assert_size_stride(arg38_1, (448, ), (1, ))
    assert_size_stride(arg39_1, (448, ), (1, ))
    assert_size_stride(arg40_1, (448, ), (1, ))
    assert_size_stride(arg41_1, (448, ), (1, ))
    assert_size_stride(arg42_1, (448, ), (1, ))
    assert_size_stride(arg43_1, (448, ), (1, ))
    assert_size_stride(arg44_1, (448, ), (1, ))
    assert_size_stride(arg45_1, (448, ), (1, ))
    assert_size_stride(arg46_1, (448, ), (1, ))
    assert_size_stride(arg47_1, (448, ), (1, ))
    assert_size_stride(arg48_1, (896, ), (1, ))
    assert_size_stride(arg49_1, (896, ), (1, ))
    assert_size_stride(arg50_1, (896, ), (1, ))
    assert_size_stride(arg51_1, (896, ), (1, ))
    assert_size_stride(arg52_1, (896, ), (1, ))
    assert_size_stride(arg53_1, (896, ), (1, ))
    assert_size_stride(arg54_1, (896, ), (1, ))
    assert_size_stride(arg55_1, (896, ), (1, ))
    assert_size_stride(arg56_1, (896, ), (1, ))
    assert_size_stride(arg57_1, (896, ), (1, ))
    assert_size_stride(arg58_1, (896, ), (1, ))
    assert_size_stride(arg59_1, (896, ), (1, ))
    assert_size_stride(arg60_1, (896, ), (1, ))
    assert_size_stride(arg61_1, (896, ), (1, ))
    assert_size_stride(arg62_1, (896, ), (1, ))
    assert_size_stride(arg63_1, (896, ), (1, ))
    assert_size_stride(arg64_1, (896, ), (1, ))
    assert_size_stride(arg65_1, (896, ), (1, ))
    assert_size_stride(arg66_1, (896, ), (1, ))
    assert_size_stride(arg67_1, (896, ), (1, ))
    assert_size_stride(arg68_1, (896, ), (1, ))
    assert_size_stride(arg69_1, (896, ), (1, ))
    assert_size_stride(arg70_1, (896, ), (1, ))
    assert_size_stride(arg71_1, (896, ), (1, ))
    assert_size_stride(arg72_1, (896, ), (1, ))
    assert_size_stride(arg73_1, (896, ), (1, ))
    assert_size_stride(arg74_1, (896, ), (1, ))
    assert_size_stride(arg75_1, (896, ), (1, ))
    assert_size_stride(arg76_1, (896, ), (1, ))
    assert_size_stride(arg77_1, (896, ), (1, ))
    assert_size_stride(arg78_1, (896, ), (1, ))
    assert_size_stride(arg79_1, (896, ), (1, ))
    assert_size_stride(arg80_1, (896, ), (1, ))
    assert_size_stride(arg81_1, (896, ), (1, ))
    assert_size_stride(arg82_1, (896, ), (1, ))
    assert_size_stride(arg83_1, (896, ), (1, ))
    assert_size_stride(arg84_1, (896, ), (1, ))
    assert_size_stride(arg85_1, (896, ), (1, ))
    assert_size_stride(arg86_1, (896, ), (1, ))
    assert_size_stride(arg87_1, (896, ), (1, ))
    assert_size_stride(arg88_1, (896, ), (1, ))
    assert_size_stride(arg89_1, (896, ), (1, ))
    assert_size_stride(arg90_1, (896, ), (1, ))
    assert_size_stride(arg91_1, (896, ), (1, ))
    assert_size_stride(arg92_1, (896, ), (1, ))
    assert_size_stride(arg93_1, (896, ), (1, ))
    assert_size_stride(arg94_1, (896, ), (1, ))
    assert_size_stride(arg95_1, (896, ), (1, ))
    assert_size_stride(arg96_1, (896, ), (1, ))
    assert_size_stride(arg97_1, (896, ), (1, ))
    assert_size_stride(arg98_1, (896, ), (1, ))
    assert_size_stride(arg99_1, (896, ), (1, ))
    assert_size_stride(arg100_1, (896, ), (1, ))
    assert_size_stride(arg101_1, (896, ), (1, ))
    assert_size_stride(arg102_1, (896, ), (1, ))
    assert_size_stride(arg103_1, (896, ), (1, ))
    assert_size_stride(arg104_1, (896, ), (1, ))
    assert_size_stride(arg105_1, (896, ), (1, ))
    assert_size_stride(arg106_1, (896, ), (1, ))
    assert_size_stride(arg107_1, (896, ), (1, ))
    assert_size_stride(arg108_1, (896, ), (1, ))
    assert_size_stride(arg109_1, (896, ), (1, ))
    assert_size_stride(arg110_1, (896, ), (1, ))
    assert_size_stride(arg111_1, (896, ), (1, ))
    assert_size_stride(arg112_1, (896, ), (1, ))
    assert_size_stride(arg113_1, (896, ), (1, ))
    assert_size_stride(arg114_1, (896, ), (1, ))
    assert_size_stride(arg115_1, (896, ), (1, ))
    assert_size_stride(arg116_1, (2240, ), (1, ))
    assert_size_stride(arg117_1, (2240, ), (1, ))
    assert_size_stride(arg118_1, (2240, ), (1, ))
    assert_size_stride(arg119_1, (2240, ), (1, ))
    assert_size_stride(arg120_1, (2240, ), (1, ))
    assert_size_stride(arg121_1, (2240, ), (1, ))
    assert_size_stride(arg122_1, (2240, ), (1, ))
    assert_size_stride(arg123_1, (2240, ), (1, ))
    assert_size_stride(arg124_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg125_1, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg126_1, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg127_1, (8, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg128_1, (8, ), (1, ))
    assert_size_stride(arg129_1, (224, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg130_1, (224, ), (1, ))
    assert_size_stride(arg131_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg132_1, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg133_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg134_1, (224, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg135_1, (56, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg136_1, (56, ), (1, ))
    assert_size_stride(arg137_1, (224, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg138_1, (224, ), (1, ))
    assert_size_stride(arg139_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg140_1, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg141_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg142_1, (56, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg143_1, (56, ), (1, ))
    assert_size_stride(arg144_1, (448, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg145_1, (448, ), (1, ))
    assert_size_stride(arg146_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg147_1, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg148_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg149_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg150_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg151_1, (112, ), (1, ))
    assert_size_stride(arg152_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg153_1, (448, ), (1, ))
    assert_size_stride(arg154_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg155_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg156_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg157_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg158_1, (112, ), (1, ))
    assert_size_stride(arg159_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg160_1, (448, ), (1, ))
    assert_size_stride(arg161_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg162_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg163_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg164_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg165_1, (112, ), (1, ))
    assert_size_stride(arg166_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg167_1, (448, ), (1, ))
    assert_size_stride(arg168_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg169_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg170_1, (448, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg171_1, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg172_1, (112, ), (1, ))
    assert_size_stride(arg173_1, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg174_1, (448, ), (1, ))
    assert_size_stride(arg175_1, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg176_1, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg177_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg178_1, (112, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg179_1, (112, ), (1, ))
    assert_size_stride(arg180_1, (896, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg181_1, (896, ), (1, ))
    assert_size_stride(arg182_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg183_1, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg184_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg185_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg186_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg187_1, (224, ), (1, ))
    assert_size_stride(arg188_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg189_1, (896, ), (1, ))
    assert_size_stride(arg190_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg191_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg192_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg193_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg194_1, (224, ), (1, ))
    assert_size_stride(arg195_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg196_1, (896, ), (1, ))
    assert_size_stride(arg197_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg198_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg199_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg200_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg201_1, (224, ), (1, ))
    assert_size_stride(arg202_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg203_1, (896, ), (1, ))
    assert_size_stride(arg204_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg205_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg206_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg207_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg208_1, (224, ), (1, ))
    assert_size_stride(arg209_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg210_1, (896, ), (1, ))
    assert_size_stride(arg211_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg212_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg213_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg214_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg215_1, (224, ), (1, ))
    assert_size_stride(arg216_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg217_1, (896, ), (1, ))
    assert_size_stride(arg218_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg219_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg220_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg221_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg222_1, (224, ), (1, ))
    assert_size_stride(arg223_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg224_1, (896, ), (1, ))
    assert_size_stride(arg225_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg226_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg227_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg228_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg229_1, (224, ), (1, ))
    assert_size_stride(arg230_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg231_1, (896, ), (1, ))
    assert_size_stride(arg232_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg233_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg234_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg235_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg236_1, (224, ), (1, ))
    assert_size_stride(arg237_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg238_1, (896, ), (1, ))
    assert_size_stride(arg239_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg240_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg241_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg242_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg243_1, (224, ), (1, ))
    assert_size_stride(arg244_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg245_1, (896, ), (1, ))
    assert_size_stride(arg246_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg247_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg248_1, (896, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg249_1, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg250_1, (224, ), (1, ))
    assert_size_stride(arg251_1, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg252_1, (896, ), (1, ))
    assert_size_stride(arg253_1, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg254_1, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg255_1, (2240, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg256_1, (224, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg257_1, (224, ), (1, ))
    assert_size_stride(arg258_1, (2240, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg259_1, (2240, ), (1, ))
    assert_size_stride(arg260_1, (2240, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg261_1, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg262_1, (1000, 2240), (2240, 1))
    assert_size_stride(arg263_1, (1000, ), (1, ))
    assert_size_stride(arg264_1, (32, ), (1, ))
    assert_size_stride(arg265_1, (32, ), (1, ))
    assert_size_stride(arg266_1, (224, ), (1, ))
    assert_size_stride(arg267_1, (224, ), (1, ))
    assert_size_stride(arg268_1, (224, ), (1, ))
    assert_size_stride(arg269_1, (224, ), (1, ))
    assert_size_stride(arg270_1, (224, ), (1, ))
    assert_size_stride(arg271_1, (224, ), (1, ))
    assert_size_stride(arg272_1, (224, ), (1, ))
    assert_size_stride(arg273_1, (224, ), (1, ))
    assert_size_stride(arg274_1, (224, ), (1, ))
    assert_size_stride(arg275_1, (224, ), (1, ))
    assert_size_stride(arg276_1, (224, ), (1, ))
    assert_size_stride(arg277_1, (224, ), (1, ))
    assert_size_stride(arg278_1, (224, ), (1, ))
    assert_size_stride(arg279_1, (224, ), (1, ))
    assert_size_stride(arg280_1, (448, ), (1, ))
    assert_size_stride(arg281_1, (448, ), (1, ))
    assert_size_stride(arg282_1, (448, ), (1, ))
    assert_size_stride(arg283_1, (448, ), (1, ))
    assert_size_stride(arg284_1, (448, ), (1, ))
    assert_size_stride(arg285_1, (448, ), (1, ))
    assert_size_stride(arg286_1, (448, ), (1, ))
    assert_size_stride(arg287_1, (448, ), (1, ))
    assert_size_stride(arg288_1, (448, ), (1, ))
    assert_size_stride(arg289_1, (448, ), (1, ))
    assert_size_stride(arg290_1, (448, ), (1, ))
    assert_size_stride(arg291_1, (448, ), (1, ))
    assert_size_stride(arg292_1, (448, ), (1, ))
    assert_size_stride(arg293_1, (448, ), (1, ))
    assert_size_stride(arg294_1, (448, ), (1, ))
    assert_size_stride(arg295_1, (448, ), (1, ))
    assert_size_stride(arg296_1, (448, ), (1, ))
    assert_size_stride(arg297_1, (448, ), (1, ))
    assert_size_stride(arg298_1, (448, ), (1, ))
    assert_size_stride(arg299_1, (448, ), (1, ))
    assert_size_stride(arg300_1, (448, ), (1, ))
    assert_size_stride(arg301_1, (448, ), (1, ))
    assert_size_stride(arg302_1, (448, ), (1, ))
    assert_size_stride(arg303_1, (448, ), (1, ))
    assert_size_stride(arg304_1, (448, ), (1, ))
    assert_size_stride(arg305_1, (448, ), (1, ))
    assert_size_stride(arg306_1, (448, ), (1, ))
    assert_size_stride(arg307_1, (448, ), (1, ))
    assert_size_stride(arg308_1, (448, ), (1, ))
    assert_size_stride(arg309_1, (448, ), (1, ))
    assert_size_stride(arg310_1, (448, ), (1, ))
    assert_size_stride(arg311_1, (448, ), (1, ))
    assert_size_stride(arg312_1, (896, ), (1, ))
    assert_size_stride(arg313_1, (896, ), (1, ))
    assert_size_stride(arg314_1, (896, ), (1, ))
    assert_size_stride(arg315_1, (896, ), (1, ))
    assert_size_stride(arg316_1, (896, ), (1, ))
    assert_size_stride(arg317_1, (896, ), (1, ))
    assert_size_stride(arg318_1, (896, ), (1, ))
    assert_size_stride(arg319_1, (896, ), (1, ))
    assert_size_stride(arg320_1, (896, ), (1, ))
    assert_size_stride(arg321_1, (896, ), (1, ))
    assert_size_stride(arg322_1, (896, ), (1, ))
    assert_size_stride(arg323_1, (896, ), (1, ))
    assert_size_stride(arg324_1, (896, ), (1, ))
    assert_size_stride(arg325_1, (896, ), (1, ))
    assert_size_stride(arg326_1, (896, ), (1, ))
    assert_size_stride(arg327_1, (896, ), (1, ))
    assert_size_stride(arg328_1, (896, ), (1, ))
    assert_size_stride(arg329_1, (896, ), (1, ))
    assert_size_stride(arg330_1, (896, ), (1, ))
    assert_size_stride(arg331_1, (896, ), (1, ))
    assert_size_stride(arg332_1, (896, ), (1, ))
    assert_size_stride(arg333_1, (896, ), (1, ))
    assert_size_stride(arg334_1, (896, ), (1, ))
    assert_size_stride(arg335_1, (896, ), (1, ))
    assert_size_stride(arg336_1, (896, ), (1, ))
    assert_size_stride(arg337_1, (896, ), (1, ))
    assert_size_stride(arg338_1, (896, ), (1, ))
    assert_size_stride(arg339_1, (896, ), (1, ))
    assert_size_stride(arg340_1, (896, ), (1, ))
    assert_size_stride(arg341_1, (896, ), (1, ))
    assert_size_stride(arg342_1, (896, ), (1, ))
    assert_size_stride(arg343_1, (896, ), (1, ))
    assert_size_stride(arg344_1, (896, ), (1, ))
    assert_size_stride(arg345_1, (896, ), (1, ))
    assert_size_stride(arg346_1, (896, ), (1, ))
    assert_size_stride(arg347_1, (896, ), (1, ))
    assert_size_stride(arg348_1, (896, ), (1, ))
    assert_size_stride(arg349_1, (896, ), (1, ))
    assert_size_stride(arg350_1, (896, ), (1, ))
    assert_size_stride(arg351_1, (896, ), (1, ))
    assert_size_stride(arg352_1, (896, ), (1, ))
    assert_size_stride(arg353_1, (896, ), (1, ))
    assert_size_stride(arg354_1, (896, ), (1, ))
    assert_size_stride(arg355_1, (896, ), (1, ))
    assert_size_stride(arg356_1, (896, ), (1, ))
    assert_size_stride(arg357_1, (896, ), (1, ))
    assert_size_stride(arg358_1, (896, ), (1, ))
    assert_size_stride(arg359_1, (896, ), (1, ))
    assert_size_stride(arg360_1, (896, ), (1, ))
    assert_size_stride(arg361_1, (896, ), (1, ))
    assert_size_stride(arg362_1, (896, ), (1, ))
    assert_size_stride(arg363_1, (896, ), (1, ))
    assert_size_stride(arg364_1, (896, ), (1, ))
    assert_size_stride(arg365_1, (896, ), (1, ))
    assert_size_stride(arg366_1, (896, ), (1, ))
    assert_size_stride(arg367_1, (896, ), (1, ))
    assert_size_stride(arg368_1, (896, ), (1, ))
    assert_size_stride(arg369_1, (896, ), (1, ))
    assert_size_stride(arg370_1, (896, ), (1, ))
    assert_size_stride(arg371_1, (896, ), (1, ))
    assert_size_stride(arg372_1, (896, ), (1, ))
    assert_size_stride(arg373_1, (896, ), (1, ))
    assert_size_stride(arg374_1, (896, ), (1, ))
    assert_size_stride(arg375_1, (896, ), (1, ))
    assert_size_stride(arg376_1, (896, ), (1, ))
    assert_size_stride(arg377_1, (896, ), (1, ))
    assert_size_stride(arg378_1, (896, ), (1, ))
    assert_size_stride(arg379_1, (896, ), (1, ))
    assert_size_stride(arg380_1, (2240, ), (1, ))
    assert_size_stride(arg381_1, (2240, ), (1, ))
    assert_size_stride(arg382_1, (2240, ), (1, ))
    assert_size_stride(arg383_1, (2240, ), (1, ))
    assert_size_stride(arg384_1, (2240, ), (1, ))
    assert_size_stride(arg385_1, (2240, ), (1, ))
    assert_size_stride(arg386_1, (2240, ), (1, ))
    assert_size_stride(arg387_1, (2240, ), (1, ))
    assert_size_stride(arg388_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg388_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg124_1
    del arg388_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg264_1
    del arg265_1
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf4, (4, 224, 112, 112), (2809856, 1, 25088, 224))
    del arg125_1
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((224, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg126_1
    del arg266_1
    del arg267_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf7 = extern_kernels.convolution(buf5, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf7, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del buf5
    buf8 = buf7; del buf7  # reuse
    buf9 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf9, (4, 224, 1, 1), (224, 1, 224, 224), 0); del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_3(c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg268_1
    del arg269_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
    buf11 = extern_kernels.convolution(buf10, arg127_1, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (4, 8, 1, 1), (8, 1, 8, 8))
    del arg127_1
    del arg128_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    cpp_fused_relu_4(c_void_p(buf12.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.relu]
    buf13 = extern_kernels.convolution(buf12, arg129_1, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf13, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg129_1
    del arg130_1
    del buf12
    buf14 = buf8; del buf8  # reuse
    cpp_fused_mul_sigmoid_5(c_void_p(buf14.data_ptr()), c_void_p(buf13.data_ptr()))
    # Source Nodes: [sigmoid, x_18, x_19], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf15 = extern_kernels.convolution(buf14, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del arg131_1
    del buf14
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf3, arg132_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del arg132_1
    del buf3
    buf17 = buf15; del buf15  # reuse
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_6(c_void_p(buf18.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg270_1
    del arg271_1
    del arg272_1
    del arg273_1
    del arg6_1
    del arg7_1
    del arg8_1
    del arg9_1
    del buf16
    # Source Nodes: [shortcut_1, x_33], Original ATen: [aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf18, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del arg133_1
    buf20 = buf19; del buf19  # reuse
    buf21 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7(c_void_p(buf20.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg10_1
    del arg11_1
    del arg134_1
    del arg274_1
    del arg275_1
    # Source Nodes: [x_34, x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf22, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del buf20
    del buf21
    buf23 = buf22; del buf22  # reuse
    buf24 = reinterpret_tensor(buf13, (4, 224, 1, 1), (224, 1, 896, 896), 0); del buf13  # reuse
    buf25 = reinterpret_tensor(buf24, (4, 224, 1, 1), (224, 1, 224, 224), 0); del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_8(c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg276_1
    del arg277_1
    # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
    buf26 = extern_kernels.convolution(buf25, arg135_1, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (4, 56, 1, 1), (56, 1, 56, 56))
    del arg135_1
    del arg136_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    cpp_fused_relu_9(c_void_p(buf27.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.relu]
    buf28 = extern_kernels.convolution(buf27, arg137_1, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf28, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg137_1
    del arg138_1
    del buf27
    buf29 = buf23; del buf23  # reuse
    cpp_fused_mul_sigmoid_10(c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()))
    del buf28
    # Source Nodes: [sigmoid_1, x_45, x_46], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf30 = extern_kernels.convolution(buf29, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (4, 224, 56, 56), (702464, 1, 12544, 224))
    del arg139_1
    del buf29
    buf31 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_11(c_void_p(buf31.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg278_1
    del arg279_1
    del buf30
    # Source Nodes: [x_56], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (4, 448, 56, 56), (1404928, 1, 25088, 448))
    del arg140_1
    buf33 = buf32; del buf32  # reuse
    buf34 = empty_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12(c_void_p(buf33.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf34.data_ptr()))
    del arg141_1
    del arg16_1
    del arg17_1
    del arg280_1
    del arg281_1
    # Source Nodes: [x_57, x_61, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf35 = extern_kernels.convolution(buf33, buf34, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf35, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del buf33
    buf36 = buf35; del buf35  # reuse
    buf37 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cpu', dtype=torch.float32)
    buf38 = reinterpret_tensor(buf37, (4, 448, 1, 1), (448, 1, 448, 448), 0); del buf37  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_13(c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg282_1
    del arg283_1
    # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
    buf39 = extern_kernels.convolution(buf38, arg142_1, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf39, (4, 56, 1, 1), (56, 1, 56, 56))
    del arg142_1
    del arg143_1
    del buf38
    buf40 = buf39; del buf39  # reuse
    cpp_fused_relu_14(c_void_p(buf40.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.relu]
    buf41 = extern_kernels.convolution(buf40, arg144_1, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (4, 448, 1, 1), (448, 1, 448, 448))
    del arg144_1
    del arg145_1
    del buf40
    buf42 = buf36; del buf36  # reuse
    cpp_fused_mul_sigmoid_15(c_void_p(buf42.data_ptr()), c_void_p(buf41.data_ptr()))
    # Source Nodes: [sigmoid_2, x_68, x_69], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf43 = extern_kernels.convolution(buf42, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg146_1
    del buf42
    # Source Nodes: [x_75], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf31, arg147_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg147_1
    del buf31
    buf45 = buf43; del buf43  # reuse
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf46.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg22_1
    del arg23_1
    del arg284_1
    del arg285_1
    del arg286_1
    del arg287_1
    del buf44
    # Source Nodes: [shortcut_3, x_83], Original ATen: [aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf46, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg148_1
    buf48 = buf47; del buf47  # reuse
    buf49 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf48.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg149_1
    del arg24_1
    del arg25_1
    del arg288_1
    del arg289_1
    # Source Nodes: [x_84, x_88, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf50, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del buf48
    buf51 = buf50; del buf50  # reuse
    buf52 = reinterpret_tensor(buf41, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf41  # reuse
    buf53 = reinterpret_tensor(buf52, (4, 448, 1, 1), (448, 1, 448, 448), 0); del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_18(c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    del arg290_1
    del arg291_1
    # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
    buf54 = extern_kernels.convolution(buf53, arg150_1, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf54, (4, 112, 1, 1), (112, 1, 112, 112))
    del arg150_1
    del arg151_1
    del buf53
    buf55 = buf54; del buf54  # reuse
    cpp_fused_relu_19(c_void_p(buf55.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.relu]
    buf56 = extern_kernels.convolution(buf55, arg152_1, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf56, (4, 448, 1, 1), (448, 1, 448, 448))
    del arg152_1
    del arg153_1
    del buf55
    buf57 = buf51; del buf51  # reuse
    cpp_fused_mul_sigmoid_20(c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()))
    # Source Nodes: [sigmoid_3, x_95, x_96], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf58 = extern_kernels.convolution(buf57, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg154_1
    del buf57
    buf59 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_21(c_void_p(buf59.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg28_1
    del arg292_1
    del arg293_1
    del arg29_1
    del buf58
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg155_1
    buf61 = buf60; del buf60  # reuse
    buf62 = buf49; del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22(c_void_p(buf61.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg156_1
    del arg294_1
    del arg295_1
    del arg30_1
    del arg31_1
    # Source Nodes: [x_106, x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf63 = extern_kernels.convolution(buf61, buf62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf63, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del buf61
    buf64 = buf63; del buf63  # reuse
    buf65 = reinterpret_tensor(buf56, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf56  # reuse
    buf66 = reinterpret_tensor(buf65, (4, 448, 1, 1), (448, 1, 448, 448), 0); del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_23(c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg296_1
    del arg297_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
    buf67 = extern_kernels.convolution(buf66, arg157_1, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf67, (4, 112, 1, 1), (112, 1, 112, 112))
    del arg157_1
    del arg158_1
    del buf66
    buf68 = buf67; del buf67  # reuse
    cpp_fused_relu_24(c_void_p(buf68.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.relu]
    buf69 = extern_kernels.convolution(buf68, arg159_1, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf69, (4, 448, 1, 1), (448, 1, 448, 448))
    del arg159_1
    del arg160_1
    del buf68
    buf70 = buf64; del buf64  # reuse
    cpp_fused_mul_sigmoid_25(c_void_p(buf70.data_ptr()), c_void_p(buf69.data_ptr()))
    # Source Nodes: [sigmoid_4, x_117, x_118], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf71 = extern_kernels.convolution(buf70, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg161_1
    del buf70
    buf72 = buf59; del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_26(c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg298_1
    del arg299_1
    del arg34_1
    del arg35_1
    del buf71
    # Source Nodes: [x_127], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg162_1
    buf74 = buf73; del buf73  # reuse
    buf75 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27(c_void_p(buf74.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg163_1
    del arg300_1
    del arg301_1
    del arg36_1
    del arg37_1
    # Source Nodes: [x_128, x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf76 = extern_kernels.convolution(buf74, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf76, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del buf74
    buf77 = buf76; del buf76  # reuse
    buf78 = reinterpret_tensor(buf69, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf69  # reuse
    buf79 = reinterpret_tensor(buf78, (4, 448, 1, 1), (448, 1, 448, 448), 0); del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_28(c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg302_1
    del arg303_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
    buf80 = extern_kernels.convolution(buf79, arg164_1, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf80, (4, 112, 1, 1), (112, 1, 112, 112))
    del arg164_1
    del arg165_1
    del buf79
    buf81 = buf80; del buf80  # reuse
    cpp_fused_relu_29(c_void_p(buf81.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.relu]
    buf82 = extern_kernels.convolution(buf81, arg166_1, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (4, 448, 1, 1), (448, 1, 448, 448))
    del arg166_1
    del arg167_1
    del buf81
    buf83 = buf77; del buf77  # reuse
    cpp_fused_mul_sigmoid_30(c_void_p(buf83.data_ptr()), c_void_p(buf82.data_ptr()))
    # Source Nodes: [sigmoid_5, x_139, x_140], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf84 = extern_kernels.convolution(buf83, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg168_1
    del buf83
    buf85 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_31(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg304_1
    del arg305_1
    del arg40_1
    del arg41_1
    del buf84
    # Source Nodes: [x_149], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg169_1
    buf87 = buf86; del buf86  # reuse
    buf88 = buf75; del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf87.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg170_1
    del arg306_1
    del arg307_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_150, x_154, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf89, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del buf87
    del buf88
    buf90 = buf89; del buf89  # reuse
    buf91 = reinterpret_tensor(buf82, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf82  # reuse
    buf92 = reinterpret_tensor(buf91, (4, 448, 1, 1), (448, 1, 448, 448), 0); del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_33(c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg308_1
    del arg309_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
    buf93 = extern_kernels.convolution(buf92, arg171_1, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf93, (4, 112, 1, 1), (112, 1, 112, 112))
    del arg171_1
    del arg172_1
    del buf92
    buf94 = buf93; del buf93  # reuse
    cpp_fused_relu_34(c_void_p(buf94.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.relu]
    buf95 = extern_kernels.convolution(buf94, arg173_1, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf95, (4, 448, 1, 1), (448, 1, 448, 448))
    del arg173_1
    del arg174_1
    del buf94
    buf96 = buf90; del buf90  # reuse
    cpp_fused_mul_sigmoid_35(c_void_p(buf96.data_ptr()), c_void_p(buf95.data_ptr()))
    del buf95
    # Source Nodes: [sigmoid_6, x_161, x_162], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf97 = extern_kernels.convolution(buf96, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf97, (4, 448, 28, 28), (351232, 1, 12544, 448))
    del arg175_1
    del buf96
    buf98 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_36(c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg310_1
    del arg311_1
    del arg46_1
    del arg47_1
    del buf97
    # Source Nodes: [x_172], Original ATen: [aten.convolution]
    buf99 = extern_kernels.convolution(buf98, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (4, 896, 28, 28), (702464, 1, 25088, 896))
    del arg176_1
    buf100 = buf99; del buf99  # reuse
    buf101 = empty_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37(c_void_p(buf100.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg177_1
    del arg312_1
    del arg313_1
    del arg48_1
    del arg49_1
    # Source Nodes: [x_173, x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf102 = extern_kernels.convolution(buf100, buf101, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf102, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf100
    buf103 = buf102; del buf102  # reuse
    buf104 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf104, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_38(c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg314_1
    del arg315_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
    buf106 = extern_kernels.convolution(buf105, arg178_1, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (4, 112, 1, 1), (112, 1, 112, 112))
    del arg178_1
    del arg179_1
    del buf105
    buf107 = buf106; del buf106  # reuse
    cpp_fused_relu_39(c_void_p(buf107.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.relu]
    buf108 = extern_kernels.convolution(buf107, arg180_1, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf108, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg180_1
    del arg181_1
    del buf107
    buf109 = buf103; del buf103  # reuse
    cpp_fused_mul_sigmoid_40(c_void_p(buf109.data_ptr()), c_void_p(buf108.data_ptr()))
    # Source Nodes: [sigmoid_7, x_184, x_185], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf110 = extern_kernels.convolution(buf109, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg182_1
    del buf109
    # Source Nodes: [x_191], Original ATen: [aten.convolution]
    buf111 = extern_kernels.convolution(buf98, arg183_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf111, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg183_1
    del buf98
    buf112 = buf110; del buf110  # reuse
    buf113 = buf112; del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_41(c_void_p(buf113.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg316_1
    del arg317_1
    del arg318_1
    del arg319_1
    del arg52_1
    del arg53_1
    del arg54_1
    del arg55_1
    del buf111
    # Source Nodes: [shortcut_8, x_199], Original ATen: [aten.convolution, aten.relu]
    buf114 = extern_kernels.convolution(buf113, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg184_1
    buf115 = buf114; del buf114  # reuse
    buf116 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42(c_void_p(buf115.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg185_1
    del arg320_1
    del arg321_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_200, x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf117 = extern_kernels.convolution(buf115, buf116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf117, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf115
    buf118 = buf117; del buf117  # reuse
    buf119 = reinterpret_tensor(buf108, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf108  # reuse
    buf120 = reinterpret_tensor(buf119, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_43(c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
    buf121 = extern_kernels.convolution(buf120, arg186_1, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf121, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg186_1
    del arg187_1
    del buf120
    buf122 = buf121; del buf121  # reuse
    cpp_fused_relu_44(c_void_p(buf122.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.relu]
    buf123 = extern_kernels.convolution(buf122, arg188_1, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf123, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg188_1
    del arg189_1
    del buf122
    buf124 = buf118; del buf118  # reuse
    cpp_fused_mul_sigmoid_45(c_void_p(buf124.data_ptr()), c_void_p(buf123.data_ptr()))
    # Source Nodes: [sigmoid_8, x_211, x_212], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf125 = extern_kernels.convolution(buf124, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf125, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg190_1
    del buf124
    buf126 = buf113; del buf113  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf126.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg324_1
    del arg325_1
    del arg60_1
    del arg61_1
    del buf125
    # Source Nodes: [x_221], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf126, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg191_1
    buf128 = buf127; del buf127  # reuse
    buf129 = buf116; del buf116  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_47(c_void_p(buf128.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg192_1
    del arg326_1
    del arg327_1
    del arg62_1
    del arg63_1
    # Source Nodes: [x_222, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf130 = extern_kernels.convolution(buf128, buf129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf130, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf128
    buf131 = buf130; del buf130  # reuse
    buf132 = reinterpret_tensor(buf123, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf123  # reuse
    buf133 = reinterpret_tensor(buf132, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_48(c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg328_1
    del arg329_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
    buf134 = extern_kernels.convolution(buf133, arg193_1, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf134, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg193_1
    del arg194_1
    del buf133
    buf135 = buf134; del buf134  # reuse
    cpp_fused_relu_49(c_void_p(buf135.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.relu]
    buf136 = extern_kernels.convolution(buf135, arg195_1, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf136, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg195_1
    del arg196_1
    del buf135
    buf137 = buf131; del buf131  # reuse
    cpp_fused_mul_sigmoid_50(c_void_p(buf137.data_ptr()), c_void_p(buf136.data_ptr()))
    # Source Nodes: [sigmoid_9, x_233, x_234], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf138 = extern_kernels.convolution(buf137, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf138, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg197_1
    del buf137
    buf139 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_51(c_void_p(buf139.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg330_1
    del arg331_1
    del arg66_1
    del arg67_1
    del buf138
    # Source Nodes: [x_243], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg198_1
    buf141 = buf140; del buf140  # reuse
    buf142 = buf129; del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52(c_void_p(buf141.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(buf142.data_ptr()))
    del arg199_1
    del arg332_1
    del arg333_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_244, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf143, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf141
    buf144 = buf143; del buf143  # reuse
    buf145 = reinterpret_tensor(buf136, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf136  # reuse
    buf146 = reinterpret_tensor(buf145, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf145  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_53(c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg334_1
    del arg335_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
    buf147 = extern_kernels.convolution(buf146, arg200_1, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf147, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg200_1
    del arg201_1
    del buf146
    buf148 = buf147; del buf147  # reuse
    cpp_fused_relu_54(c_void_p(buf148.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.relu]
    buf149 = extern_kernels.convolution(buf148, arg202_1, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf149, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg202_1
    del arg203_1
    del buf148
    buf150 = buf144; del buf144  # reuse
    cpp_fused_mul_sigmoid_55(c_void_p(buf150.data_ptr()), c_void_p(buf149.data_ptr()))
    # Source Nodes: [sigmoid_10, x_255, x_256], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf151 = extern_kernels.convolution(buf150, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg204_1
    del buf150
    buf152 = buf139; del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_56(c_void_p(buf152.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg336_1
    del arg337_1
    del arg72_1
    del arg73_1
    del buf151
    # Source Nodes: [x_265], Original ATen: [aten.convolution]
    buf153 = extern_kernels.convolution(buf152, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf153, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg205_1
    buf154 = buf153; del buf153  # reuse
    buf155 = buf142; del buf142  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57(c_void_p(buf154.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg206_1
    del arg338_1
    del arg339_1
    del arg74_1
    del arg75_1
    # Source Nodes: [x_266, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf156, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf154
    buf157 = buf156; del buf156  # reuse
    buf158 = reinterpret_tensor(buf149, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf149  # reuse
    buf159 = reinterpret_tensor(buf158, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf158  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_58(c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg340_1
    del arg341_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
    buf160 = extern_kernels.convolution(buf159, arg207_1, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf160, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg207_1
    del arg208_1
    del buf159
    buf161 = buf160; del buf160  # reuse
    cpp_fused_relu_59(c_void_p(buf161.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.relu]
    buf162 = extern_kernels.convolution(buf161, arg209_1, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf162, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg209_1
    del arg210_1
    del buf161
    buf163 = buf157; del buf157  # reuse
    cpp_fused_mul_sigmoid_60(c_void_p(buf163.data_ptr()), c_void_p(buf162.data_ptr()))
    # Source Nodes: [sigmoid_11, x_277, x_278], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf164 = extern_kernels.convolution(buf163, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf164, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg211_1
    del buf163
    buf165 = buf152; del buf152  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_61(c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg342_1
    del arg343_1
    del arg78_1
    del arg79_1
    del buf164
    # Source Nodes: [x_287], Original ATen: [aten.convolution]
    buf166 = extern_kernels.convolution(buf165, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf166, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg212_1
    buf167 = buf166; del buf166  # reuse
    buf168 = buf155; del buf155  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_62(c_void_p(buf167.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf168.data_ptr()))
    del arg213_1
    del arg344_1
    del arg345_1
    del arg80_1
    del arg81_1
    # Source Nodes: [x_288, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf169 = extern_kernels.convolution(buf167, buf168, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf169, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf167
    buf170 = buf169; del buf169  # reuse
    buf171 = reinterpret_tensor(buf162, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf162  # reuse
    buf172 = reinterpret_tensor(buf171, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf171  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_63(c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg346_1
    del arg347_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean]
    buf173 = extern_kernels.convolution(buf172, arg214_1, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf173, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg214_1
    del arg215_1
    del buf172
    buf174 = buf173; del buf173  # reuse
    cpp_fused_relu_64(c_void_p(buf174.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.relu]
    buf175 = extern_kernels.convolution(buf174, arg216_1, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf175, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg216_1
    del arg217_1
    del buf174
    buf176 = buf170; del buf170  # reuse
    cpp_fused_mul_sigmoid_65(c_void_p(buf176.data_ptr()), c_void_p(buf175.data_ptr()))
    # Source Nodes: [sigmoid_12, x_299, x_300], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf177 = extern_kernels.convolution(buf176, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf177, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg218_1
    del buf176
    buf178 = buf165; del buf165  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_66(c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg348_1
    del arg349_1
    del arg84_1
    del arg85_1
    del buf177
    # Source Nodes: [x_309], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf178, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf179, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg219_1
    buf180 = buf179; del buf179  # reuse
    buf181 = buf168; del buf168  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_67(c_void_p(buf180.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg220_1
    del arg350_1
    del arg351_1
    del arg86_1
    del arg87_1
    # Source Nodes: [x_310, x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf182, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf180
    buf183 = buf182; del buf182  # reuse
    buf184 = reinterpret_tensor(buf175, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf175  # reuse
    buf185 = reinterpret_tensor(buf184, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf184  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_68(c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg352_1
    del arg353_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean]
    buf186 = extern_kernels.convolution(buf185, arg221_1, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf186, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg221_1
    del arg222_1
    del buf185
    buf187 = buf186; del buf186  # reuse
    cpp_fused_relu_69(c_void_p(buf187.data_ptr()))
    # Source Nodes: [x_se_54, x_se_55], Original ATen: [aten.convolution, aten.relu]
    buf188 = extern_kernels.convolution(buf187, arg223_1, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf188, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg223_1
    del arg224_1
    del buf187
    buf189 = buf183; del buf183  # reuse
    cpp_fused_mul_sigmoid_70(c_void_p(buf189.data_ptr()), c_void_p(buf188.data_ptr()))
    # Source Nodes: [sigmoid_13, x_321, x_322], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf190 = extern_kernels.convolution(buf189, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf190, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg225_1
    del buf189
    buf191 = buf178; del buf178  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_71(c_void_p(buf191.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg354_1
    del arg355_1
    del arg90_1
    del arg91_1
    del buf190
    # Source Nodes: [x_331], Original ATen: [aten.convolution]
    buf192 = extern_kernels.convolution(buf191, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf192, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg226_1
    buf193 = buf192; del buf192  # reuse
    buf194 = buf181; del buf181  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72(c_void_p(buf193.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg227_1
    del arg356_1
    del arg357_1
    del arg92_1
    del arg93_1
    # Source Nodes: [x_332, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf195 = extern_kernels.convolution(buf193, buf194, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf195, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf193
    buf196 = buf195; del buf195  # reuse
    buf197 = reinterpret_tensor(buf188, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf188  # reuse
    buf198 = reinterpret_tensor(buf197, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf197  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_73(c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg358_1
    del arg359_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean]
    buf199 = extern_kernels.convolution(buf198, arg228_1, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf199, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg228_1
    del arg229_1
    del buf198
    buf200 = buf199; del buf199  # reuse
    cpp_fused_relu_74(c_void_p(buf200.data_ptr()))
    # Source Nodes: [x_se_58, x_se_59], Original ATen: [aten.convolution, aten.relu]
    buf201 = extern_kernels.convolution(buf200, arg230_1, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf201, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg230_1
    del arg231_1
    del buf200
    buf202 = buf196; del buf196  # reuse
    cpp_fused_mul_sigmoid_75(c_void_p(buf202.data_ptr()), c_void_p(buf201.data_ptr()))
    # Source Nodes: [sigmoid_14, x_343, x_344], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf203 = extern_kernels.convolution(buf202, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg232_1
    del buf202
    buf204 = buf191; del buf191  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_76(c_void_p(buf204.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg360_1
    del arg361_1
    del arg96_1
    del arg97_1
    del buf203
    # Source Nodes: [x_353], Original ATen: [aten.convolution]
    buf205 = extern_kernels.convolution(buf204, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf205, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg233_1
    buf206 = buf205; del buf205  # reuse
    buf207 = buf194; del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_77(c_void_p(buf206.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg234_1
    del arg362_1
    del arg363_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_354, x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf208 = extern_kernels.convolution(buf206, buf207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf208, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf206
    buf209 = buf208; del buf208  # reuse
    buf210 = reinterpret_tensor(buf201, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf201  # reuse
    buf211 = reinterpret_tensor(buf210, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_78(c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg364_1
    del arg365_1
    # Source Nodes: [x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean]
    buf212 = extern_kernels.convolution(buf211, arg235_1, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf212, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg235_1
    del arg236_1
    del buf211
    buf213 = buf212; del buf212  # reuse
    cpp_fused_relu_79(c_void_p(buf213.data_ptr()))
    # Source Nodes: [x_se_62, x_se_63], Original ATen: [aten.convolution, aten.relu]
    buf214 = extern_kernels.convolution(buf213, arg237_1, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf214, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg237_1
    del arg238_1
    del buf213
    buf215 = buf209; del buf209  # reuse
    cpp_fused_mul_sigmoid_80(c_void_p(buf215.data_ptr()), c_void_p(buf214.data_ptr()))
    # Source Nodes: [sigmoid_15, x_365, x_366], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf216 = extern_kernels.convolution(buf215, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf216, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg239_1
    del buf215
    buf217 = buf204; del buf204  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_81(c_void_p(buf217.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg366_1
    del arg367_1
    del buf216
    # Source Nodes: [x_375], Original ATen: [aten.convolution]
    buf218 = extern_kernels.convolution(buf217, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf218, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg240_1
    buf219 = buf218; del buf218  # reuse
    buf220 = buf207; del buf207  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_82(c_void_p(buf219.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(buf220.data_ptr()))
    del arg104_1
    del arg105_1
    del arg241_1
    del arg368_1
    del arg369_1
    # Source Nodes: [x_376, x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf221 = extern_kernels.convolution(buf219, buf220, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf221, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf219
    buf222 = buf221; del buf221  # reuse
    buf223 = reinterpret_tensor(buf214, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf214  # reuse
    buf224 = reinterpret_tensor(buf223, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf223  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_83(c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg370_1
    del arg371_1
    # Source Nodes: [x_se_64, x_se_65], Original ATen: [aten.convolution, aten.mean]
    buf225 = extern_kernels.convolution(buf224, arg242_1, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf225, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg242_1
    del arg243_1
    del buf224
    buf226 = buf225; del buf225  # reuse
    cpp_fused_relu_84(c_void_p(buf226.data_ptr()))
    # Source Nodes: [x_se_66, x_se_67], Original ATen: [aten.convolution, aten.relu]
    buf227 = extern_kernels.convolution(buf226, arg244_1, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf227, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg244_1
    del arg245_1
    del buf226
    buf228 = buf222; del buf222  # reuse
    cpp_fused_mul_sigmoid_85(c_void_p(buf228.data_ptr()), c_void_p(buf227.data_ptr()))
    # Source Nodes: [sigmoid_16, x_387, x_388], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf229 = extern_kernels.convolution(buf228, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf229, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg246_1
    del buf228
    buf230 = buf217; del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_86(c_void_p(buf230.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg372_1
    del arg373_1
    del buf229
    # Source Nodes: [x_397], Original ATen: [aten.convolution]
    buf231 = extern_kernels.convolution(buf230, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf231, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg247_1
    buf232 = buf231; del buf231  # reuse
    buf233 = buf220; del buf220  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_87(c_void_p(buf232.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg110_1
    del arg111_1
    del arg248_1
    del arg374_1
    del arg375_1
    # Source Nodes: [x_398, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf234, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del buf232
    del buf233
    buf235 = buf234; del buf234  # reuse
    buf236 = reinterpret_tensor(buf227, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf227  # reuse
    buf237 = reinterpret_tensor(buf236, (4, 896, 1, 1), (896, 1, 896, 896), 0); del buf236  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_88(c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg376_1
    del arg377_1
    # Source Nodes: [x_se_68, x_se_69], Original ATen: [aten.convolution, aten.mean]
    buf238 = extern_kernels.convolution(buf237, arg249_1, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf238, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg249_1
    del arg250_1
    del buf237
    buf239 = buf238; del buf238  # reuse
    cpp_fused_relu_89(c_void_p(buf239.data_ptr()))
    # Source Nodes: [x_se_70, x_se_71], Original ATen: [aten.convolution, aten.relu]
    buf240 = extern_kernels.convolution(buf239, arg251_1, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf240, (4, 896, 1, 1), (896, 1, 896, 896))
    del arg251_1
    del arg252_1
    del buf239
    buf241 = buf235; del buf235  # reuse
    cpp_fused_mul_sigmoid_90(c_void_p(buf241.data_ptr()), c_void_p(buf240.data_ptr()))
    del buf240
    # Source Nodes: [sigmoid_17, x_409, x_410], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf242 = extern_kernels.convolution(buf241, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (4, 896, 14, 14), (175616, 1, 12544, 896))
    del arg253_1
    del buf241
    buf243 = buf230; del buf230  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_91(c_void_p(buf243.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg378_1
    del arg379_1
    del buf242
    # Source Nodes: [x_420], Original ATen: [aten.convolution]
    buf244 = extern_kernels.convolution(buf243, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf244, (4, 2240, 14, 14), (439040, 1, 31360, 2240))
    del arg254_1
    buf245 = buf244; del buf244  # reuse
    buf246 = empty_strided((2240, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_92(c_void_p(buf245.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(buf246.data_ptr()))
    del arg116_1
    del arg117_1
    del arg255_1
    del arg380_1
    del arg381_1
    # Source Nodes: [x_421, x_425, x_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf247 = extern_kernels.convolution(buf245, buf246, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
    assert_size_stride(buf247, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    del buf245
    del buf246
    buf248 = buf247; del buf247  # reuse
    buf249 = empty_strided((4, 2240, 1, 1), (2240, 1, 8960, 8960), device='cpu', dtype=torch.float32)
    buf250 = reinterpret_tensor(buf249, (4, 2240, 1, 1), (2240, 1, 2240, 2240), 0); del buf249  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_93(c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg382_1
    del arg383_1
    # Source Nodes: [x_se_72, x_se_73], Original ATen: [aten.convolution, aten.mean]
    buf251 = extern_kernels.convolution(buf250, arg256_1, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf251, (4, 224, 1, 1), (224, 1, 224, 224))
    del arg256_1
    del arg257_1
    del buf250
    buf252 = buf251; del buf251  # reuse
    cpp_fused_relu_94(c_void_p(buf252.data_ptr()))
    # Source Nodes: [x_se_74, x_se_75], Original ATen: [aten.convolution, aten.relu]
    buf253 = extern_kernels.convolution(buf252, arg258_1, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf253, (4, 2240, 1, 1), (2240, 1, 2240, 2240))
    del arg258_1
    del arg259_1
    del buf252
    buf254 = buf248; del buf248  # reuse
    cpp_fused_mul_sigmoid_95(c_void_p(buf254.data_ptr()), c_void_p(buf253.data_ptr()))
    # Source Nodes: [sigmoid_18, x_432, x_433], Original ATen: [aten.convolution, aten.mul, aten.sigmoid]
    buf255 = extern_kernels.convolution(buf254, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf255, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    del arg260_1
    del buf254
    # Source Nodes: [x_439], Original ATen: [aten.convolution]
    buf256 = extern_kernels.convolution(buf243, arg261_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf256, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    del arg261_1
    del buf243
    buf257 = buf255; del buf255  # reuse
    buf258 = reinterpret_tensor(buf253, (4, 2240, 1, 1), (2240, 1, 8960, 8960), 0); del buf253  # reuse
    buf259 = reinterpret_tensor(buf258, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf258  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_96(c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg122_1
    del arg123_1
    del arg384_1
    del arg385_1
    del arg386_1
    del arg387_1
    del buf256
    del buf257
    buf260 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_454], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg263_1, reinterpret_tensor(buf259, (4, 2240), (2240, 1), 0), reinterpret_tensor(arg262_1, (2240, 1000), (1, 2240), 0), alpha=1, beta=1, out=buf260)
    del arg262_1
    del arg263_1
    return (buf260, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((8, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((224, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((224, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((56, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((224, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((56, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((448, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((448, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((112, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((896, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((896, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((2240, 112, 3, 3), (1008, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((224, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((2240, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2240, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1000, 2240), (2240, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_regnet', benchmark_compiled_module)
