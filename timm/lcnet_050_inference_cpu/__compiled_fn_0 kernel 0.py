
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_1 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_3 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_4 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_6 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_7 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_8 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_9 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_10 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_11 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_13 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_14 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_15 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_16 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_17 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_18 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_19 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_20 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_21 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_23 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (6272L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        auto tmp11 = tmp10 / tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(49.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0)));
                    auto tmp1 = static_cast<float>(3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = at::vec::maximum(tmp3, tmp5);
                    auto tmp7 = static_cast<float>(6.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = at::vec::minimum(tmp6, tmp8);
                    auto tmp10 = tmp0 * tmp9;
                    auto tmp11 = tmp10 / tmp8;
                    auto tmp13 = tmp12 + tmp2;
                    auto tmp14 = at::vec::maximum(tmp13, tmp5);
                    auto tmp15 = at::vec::minimum(tmp14, tmp8);
                    auto tmp16 = tmp15 / tmp8;
                    auto tmp17 = tmp11 * tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        auto tmp11 = tmp10 / tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(49.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardswish_mul_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp1 = static_cast<float>(3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = at::vec::maximum(tmp3, tmp5);
                    auto tmp7 = static_cast<float>(6.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = at::vec::minimum(tmp6, tmp8);
                    auto tmp10 = tmp0 * tmp9;
                    auto tmp11 = tmp10 / tmp8;
                    auto tmp13 = tmp12 + tmp2;
                    auto tmp14 = at::vec::maximum(tmp13, tmp5);
                    auto tmp15 = at::vec::minimum(tmp14, tmp8);
                    auto tmp16 = tmp15 / tmp8;
                    auto tmp17 = tmp11 * tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        auto tmp11 = tmp10 / tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(49.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardswish_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, ), (1, ))
    assert_size_stride(arg1_1, (8, ), (1, ))
    assert_size_stride(arg2_1, (8, ), (1, ))
    assert_size_stride(arg3_1, (8, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (32, ), (1, ))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg55_1, (1000, ), (1, ))
    assert_size_stride(arg56_1, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg57_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg58_1, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg59_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg60_1, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg61_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg62_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg63_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg64_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg65_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg66_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg67_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg68_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg69_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg70_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg71_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg72_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg73_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg74_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg75_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg76_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg77_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg78_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg79_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg80_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg81_1, (32, ), (1, ))
    assert_size_stride(arg82_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg85_1, (256, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg86_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg87_1, (64, ), (1, ))
    assert_size_stride(arg88_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg92_1, (1280, ), (1, ))
    assert_size_stride(arg93_1, (8, ), (1, ))
    assert_size_stride(arg94_1, (8, ), (1, ))
    assert_size_stride(arg95_1, (8, ), (1, ))
    assert_size_stride(arg96_1, (8, ), (1, ))
    assert_size_stride(arg97_1, (16, ), (1, ))
    assert_size_stride(arg98_1, (16, ), (1, ))
    assert_size_stride(arg99_1, (16, ), (1, ))
    assert_size_stride(arg100_1, (16, ), (1, ))
    assert_size_stride(arg101_1, (32, ), (1, ))
    assert_size_stride(arg102_1, (32, ), (1, ))
    assert_size_stride(arg103_1, (32, ), (1, ))
    assert_size_stride(arg104_1, (32, ), (1, ))
    assert_size_stride(arg105_1, (32, ), (1, ))
    assert_size_stride(arg106_1, (32, ), (1, ))
    assert_size_stride(arg107_1, (32, ), (1, ))
    assert_size_stride(arg108_1, (32, ), (1, ))
    assert_size_stride(arg109_1, (64, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (64, ), (1, ))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (64, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg147_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg147_1
    del arg56_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 8, 112, 112), (100352, 1, 896, 8))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_1(c_void_p(buf4.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg93_1
    del arg94_1
    # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.hardswish]
    buf5 = extern_kernels.convolution(buf4, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf5, (8, 8, 112, 112), (100352, 1, 896, 8))
    del arg57_1
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_2(c_void_p(buf7.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg2_1
    del arg3_1
    del arg95_1
    del arg96_1
    # Source Nodes: [x_11, x_9], Original ATen: [aten.convolution, aten.hardswish]
    buf8 = extern_kernels.convolution(buf7, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del arg58_1
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = buf9; del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_3(c_void_p(buf10.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg4_1
    del arg5_1
    del arg97_1
    del arg98_1
    # Source Nodes: [shortcut_1, x_16], Original ATen: [aten.convolution, aten.hardswish]
    buf11 = extern_kernels.convolution(buf10, arg59_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf11, (8, 16, 56, 56), (50176, 1, 896, 16))
    del arg59_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_4(c_void_p(buf13.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg100_1
    del arg6_1
    del arg7_1
    del arg99_1
    # Source Nodes: [x_20, x_22], Original ATen: [aten.convolution, aten.hardswish]
    buf14 = extern_kernels.convolution(buf13, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del arg60_1
    del buf13
    buf15 = buf14; del buf14  # reuse
    buf16 = buf15; del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_5(c_void_p(buf16.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg101_1
    del arg102_1
    del arg8_1
    del arg9_1
    # Source Nodes: [shortcut_2, x_27], Original ATen: [aten.convolution, aten.hardswish]
    buf17 = extern_kernels.convolution(buf16, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf17, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del arg61_1
    del buf16
    buf18 = buf17; del buf17  # reuse
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_6(c_void_p(buf19.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg103_1
    del arg104_1
    del arg10_1
    del arg11_1
    # Source Nodes: [x_31, x_33], Original ATen: [aten.convolution, aten.hardswish]
    buf20 = extern_kernels.convolution(buf19, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del arg62_1
    del buf19
    buf21 = buf20; del buf20  # reuse
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_7(c_void_p(buf22.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg105_1
    del arg106_1
    del arg12_1
    del arg13_1
    # Source Nodes: [shortcut_3, x_38], Original ATen: [aten.convolution, aten.hardswish]
    buf23 = extern_kernels.convolution(buf22, arg63_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf23, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg63_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    buf25 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_8(c_void_p(buf25.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg107_1
    del arg108_1
    del arg14_1
    del arg15_1
    # Source Nodes: [x_42, x_44], Original ATen: [aten.convolution, aten.hardswish]
    buf26 = extern_kernels.convolution(buf25, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del arg64_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    buf28 = buf27; del buf27  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_9(c_void_p(buf28.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg16_1
    del arg17_1
    # Source Nodes: [shortcut_4, x_49], Original ATen: [aten.convolution, aten.hardswish]
    buf29 = extern_kernels.convolution(buf28, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf29, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del arg65_1
    del buf28
    buf30 = buf29; del buf29  # reuse
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_10(c_void_p(buf31.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg111_1
    del arg112_1
    del arg18_1
    del arg19_1
    # Source Nodes: [x_53, x_55], Original ATen: [aten.convolution, aten.hardswish]
    buf32 = extern_kernels.convolution(buf31, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del arg66_1
    del buf31
    buf33 = buf32; del buf32  # reuse
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_11(c_void_p(buf34.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg113_1
    del arg114_1
    del arg20_1
    del arg21_1
    # Source Nodes: [shortcut_5, x_60], Original ATen: [aten.convolution, aten.hardswish]
    buf35 = extern_kernels.convolution(buf34, arg67_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf35, (8, 64, 14, 14), (12544, 1, 896, 64))
    del arg67_1
    del buf34
    buf36 = buf35; del buf35  # reuse
    buf37 = buf36; del buf36  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_12(c_void_p(buf37.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg22_1
    del arg23_1
    # Source Nodes: [x_64, x_66], Original ATen: [aten.convolution, aten.hardswish]
    buf38 = extern_kernels.convolution(buf37, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg68_1
    del buf37
    buf39 = buf38; del buf38  # reuse
    buf40 = buf39; del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_13(c_void_p(buf40.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg117_1
    del arg118_1
    del arg24_1
    del arg25_1
    # Source Nodes: [shortcut_6, x_71], Original ATen: [aten.convolution, aten.hardswish]
    buf41 = extern_kernels.convolution(buf40, arg69_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf41, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg69_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    buf43 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_14(c_void_p(buf43.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg119_1
    del arg120_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_75, x_77], Original ATen: [aten.convolution, aten.hardswish]
    buf44 = extern_kernels.convolution(buf43, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg70_1
    del buf43
    buf45 = buf44; del buf44  # reuse
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_15(c_void_p(buf46.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg121_1
    del arg122_1
    del arg28_1
    del arg29_1
    # Source Nodes: [shortcut_7, x_82], Original ATen: [aten.convolution, aten.hardswish]
    buf47 = extern_kernels.convolution(buf46, arg71_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf47, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg71_1
    del buf46
    buf48 = buf47; del buf47  # reuse
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_16(c_void_p(buf49.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg123_1
    del arg124_1
    del arg30_1
    del arg31_1
    # Source Nodes: [x_86, x_88], Original ATen: [aten.convolution, aten.hardswish]
    buf50 = extern_kernels.convolution(buf49, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg72_1
    del buf49
    buf51 = buf50; del buf50  # reuse
    buf52 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_17(c_void_p(buf52.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg125_1
    del arg126_1
    del arg32_1
    del arg33_1
    # Source Nodes: [shortcut_8, x_93], Original ATen: [aten.convolution, aten.hardswish]
    buf53 = extern_kernels.convolution(buf52, arg73_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf53, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg73_1
    del buf52
    buf54 = buf53; del buf53  # reuse
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_18(c_void_p(buf55.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_97, x_99], Original ATen: [aten.convolution, aten.hardswish]
    buf56 = extern_kernels.convolution(buf55, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg74_1
    del buf55
    buf57 = buf56; del buf56  # reuse
    buf58 = buf57; del buf57  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_19(c_void_p(buf58.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg36_1
    del arg37_1
    # Source Nodes: [shortcut_9, x_104], Original ATen: [aten.convolution, aten.hardswish]
    buf59 = extern_kernels.convolution(buf58, arg75_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf59, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg75_1
    del buf58
    buf60 = buf59; del buf59  # reuse
    buf61 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_20(c_void_p(buf61.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg131_1
    del arg132_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_108, x_110], Original ATen: [aten.convolution, aten.hardswish]
    buf62 = extern_kernels.convolution(buf61, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg76_1
    del buf61
    buf63 = buf62; del buf62  # reuse
    buf64 = buf63; del buf63  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_21(c_void_p(buf64.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg40_1
    del arg41_1
    # Source Nodes: [shortcut_10, x_115], Original ATen: [aten.convolution, aten.hardswish]
    buf65 = extern_kernels.convolution(buf64, arg77_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf65, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg77_1
    del buf64
    buf66 = buf65; del buf65  # reuse
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_22(c_void_p(buf67.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg135_1
    del arg136_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_119, x_121], Original ATen: [aten.convolution, aten.hardswish]
    buf68 = extern_kernels.convolution(buf67, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del arg78_1
    del buf67
    buf69 = buf68; del buf68  # reuse
    buf70 = buf69; del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_23(c_void_p(buf70.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg137_1
    del arg138_1
    del arg44_1
    del arg45_1
    # Source Nodes: [shortcut_11, x_126], Original ATen: [aten.convolution, aten.hardswish]
    buf71 = extern_kernels.convolution(buf70, arg79_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf71, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg79_1
    del buf70
    buf72 = buf71; del buf71  # reuse
    buf73 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf73, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf73  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_24(c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_130, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf75 = extern_kernels.convolution(buf74, arg80_1, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf75, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg80_1
    del arg81_1
    del buf74
    buf76 = buf75; del buf75  # reuse
    cpp_fused_relu_25(c_void_p(buf76.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.relu]
    buf77 = extern_kernels.convolution(buf76, arg82_1, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf77, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg82_1
    del arg83_1
    del buf76
    buf78 = buf72; del buf72  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_26(c_void_p(buf78.data_ptr()), c_void_p(buf77.data_ptr()))
    del buf77
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_130, x_131, x_132], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf79 = extern_kernels.convolution(buf78, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (8, 256, 7, 7), (12544, 1, 1792, 256))
    del arg84_1
    del buf78
    buf80 = buf79; del buf79  # reuse
    buf81 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_27(c_void_p(buf81.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg141_1
    del arg142_1
    del arg48_1
    del arg49_1
    # Source Nodes: [shortcut_12, x_137], Original ATen: [aten.convolution, aten.hardswish]
    buf82 = extern_kernels.convolution(buf81, arg85_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf82, (8, 256, 7, 7), (12544, 1, 1792, 256))
    del arg85_1
    del buf81
    buf83 = buf82; del buf82  # reuse
    buf84 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf85 = reinterpret_tensor(buf84, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_28(c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg143_1
    del arg144_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_141, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf86 = extern_kernels.convolution(buf85, arg86_1, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf86, (8, 64, 1, 1), (64, 1, 64, 64))
    del arg86_1
    del arg87_1
    del buf85
    buf87 = buf86; del buf86  # reuse
    cpp_fused_relu_29(c_void_p(buf87.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.relu]
    buf88 = extern_kernels.convolution(buf87, arg88_1, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf88, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg88_1
    del arg89_1
    del buf87
    buf89 = buf83; del buf83  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_30(c_void_p(buf89.data_ptr()), c_void_p(buf88.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_141, x_142, x_143], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf90 = extern_kernels.convolution(buf89, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (8, 256, 7, 7), (12544, 1, 1792, 256))
    del arg90_1
    del buf89
    buf91 = buf90; del buf90  # reuse
    buf92 = reinterpret_tensor(buf88, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf88  # reuse
    buf93 = reinterpret_tensor(buf92, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf92  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_31(c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg52_1
    del arg53_1
    del buf91
    # Source Nodes: [x_149, x_150, x_153], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf94 = extern_kernels.convolution(buf93, arg91_1, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf94, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    del arg91_1
    del arg92_1
    del buf93
    buf95 = reinterpret_tensor(buf94, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf94  # reuse
    cpp_fused_hardswish_32(c_void_p(buf95.data_ptr()))
    buf96 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf95, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg54_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf96)
    del arg54_1
    del arg55_1
    return (buf96, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((256, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('lcnet_050', benchmark_compiled_module)
