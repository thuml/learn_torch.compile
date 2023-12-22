
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_silu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (32L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (32L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(144.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (144L*x0) + (36864L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(144.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(144.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(144.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_convolution_native_layer_norm_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(18432L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>((144L*(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 4L)) % static_cast<long>(256L))) + (36864L*x1) + (73728L*x0) + (147456L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 147456L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 1024L)) % static_cast<long>(144L)))];
                            auto tmp1 = out_ptr0[static_cast<long>((256L*x1) + (512L*x0) + (1024L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 147456L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 4L)) % static_cast<long>(256L)))];
                            auto tmp3 = out_ptr1[static_cast<long>((256L*x1) + (512L*x0) + (1024L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 147456L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 4L)) % static_cast<long>(256L)))];
                            auto tmp10 = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 1024L)) % static_cast<long>(144L))];
                            auto tmp12 = in_ptr5[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (64L*x2)), 1024L)) % static_cast<long>(144L))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(144.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            out_ptr2[static_cast<long>(x1 + (2L*x3) + (32L*x0) + (64L*x2))] = tmp13;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x3 + (32L*(static_cast<long>(x2) % static_cast<long>(2L))) + (64L*(c10::div_floor_integer((x3 + (32L*x2)), 64L))) + (1024L*x1) + (1024L*x1_inner) + (147456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr3 + static_cast<long>(x1 + (144L*x3) + (4608L*x2) + (147456L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(96);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (96L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(192);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_out_ptr0[static_cast<long>((-96L) + x1 + (96L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (192L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_silu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (32L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (32L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(192.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (192L*x0) + (12288L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(192.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(192.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(192.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(192.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(192.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(192.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_convolution_native_layer_norm_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12288L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>((192L*(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 4L)) % static_cast<long>(64L))) + (12288L*x1) + (24576L*x0) + (49152L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 49152L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 256L)) % static_cast<long>(192L)))];
                            auto tmp1 = out_ptr0[static_cast<long>((64L*x1) + (128L*x0) + (256L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 49152L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 4L)) % static_cast<long>(64L)))];
                            auto tmp3 = out_ptr1[static_cast<long>((64L*x1) + (128L*x0) + (256L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 49152L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 4L)) % static_cast<long>(64L)))];
                            auto tmp10 = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 256L)) % static_cast<long>(192L))];
                            auto tmp12 = in_ptr5[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (32L*x2)), 256L)) % static_cast<long>(192L))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            out_ptr2[static_cast<long>(x1 + (2L*x3) + (16L*x0) + (32L*x2))] = tmp13;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x3 + (16L*(static_cast<long>(x2) % static_cast<long>(2L))) + (32L*(c10::div_floor_integer((x3 + (16L*x2)), 32L))) + (256L*x1) + (256L*x1_inner) + (49152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr3 + static_cast<long>(x1 + (192L*x3) + (3072L*x2) + (49152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(256);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_out_ptr0[static_cast<long>((-128L) + x1 + (128L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_48 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_layer_norm_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (32L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (32L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(240.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (240L*x0) + (3840L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(240.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(240.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(240.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_silu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(240.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(240.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_convolution_native_layer_norm_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7680L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(4L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 4L)) % static_cast<long>(16L))) + (3840L*x1) + (7680L*x0) + (15360L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 15360L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 64L)) % static_cast<long>(240L)))];
                            auto tmp1 = in_ptr1[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 4L)) % static_cast<long>(16L))) + (3840L*x1) + (7680L*x0) + (15360L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 15360L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 64L)) % static_cast<long>(240L)))];
                            auto tmp3 = in_ptr2[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 4L)) % static_cast<long>(16L))) + (3840L*x1) + (7680L*x0) + (15360L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 15360L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 64L)) % static_cast<long>(240L)))];
                            auto tmp5 = out_ptr0[static_cast<long>((16L*x1) + (32L*x0) + (64L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 15360L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 4L)) % static_cast<long>(16L)))];
                            auto tmp7 = out_ptr1[static_cast<long>((16L*x1) + (32L*x0) + (64L*(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 15360L))) + (static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 4L)) % static_cast<long>(16L)))];
                            auto tmp14 = in_ptr3[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 64L)) % static_cast<long>(240L))];
                            auto tmp16 = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer((x1 + (2L*x0) + (4L*x3) + (16L*x2)), 64L)) % static_cast<long>(240L))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                            auto tmp8 = static_cast<float>(240.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-05);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            out_ptr2[static_cast<long>(x1 + (2L*x3) + (8L*x0) + (16L*x2))] = tmp17;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x3 + (8L*(static_cast<long>(x2) % static_cast<long>(2L))) + (16L*(c10::div_floor_integer((x3 + (8L*x2)), 16L))) + (64L*x1) + (64L*x1_inner) + (15360L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr3 + static_cast<long>(x1 + (240L*x3) + (1920L*x2) + (15360L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(160);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr4[static_cast<long>(x1 + (160L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(320);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_out_ptr0[static_cast<long>((-160L) + x1 + (160L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (320L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2880L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (320L*x2) + (2880L*x0)), static_cast<long>(320L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (2880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x2) + (40960L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (96, ), (1, ))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (96, ), (1, ))
    assert_size_stride(arg37_1, (96, ), (1, ))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (160, ), (1, ))
    assert_size_stride(arg55_1, (160, ), (1, ))
    assert_size_stride(arg56_1, (160, ), (1, ))
    assert_size_stride(arg57_1, (160, ), (1, ))
    assert_size_stride(arg58_1, (160, ), (1, ))
    assert_size_stride(arg59_1, (160, ), (1, ))
    assert_size_stride(arg60_1, (160, ), (1, ))
    assert_size_stride(arg61_1, (160, ), (1, ))
    assert_size_stride(arg62_1, (640, ), (1, ))
    assert_size_stride(arg63_1, (640, ), (1, ))
    assert_size_stride(arg64_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg65_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg66_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg68_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg69_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg70_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg71_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg72_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg74_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg75_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg76_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg78_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg79_1, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg80_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg81_1, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg82_1, (144, ), (1, ))
    assert_size_stride(arg83_1, (144, ), (1, ))
    assert_size_stride(arg84_1, (432, 144), (144, 1))
    assert_size_stride(arg85_1, (432, ), (1, ))
    assert_size_stride(arg86_1, (144, 144), (144, 1))
    assert_size_stride(arg87_1, (144, ), (1, ))
    assert_size_stride(arg88_1, (144, ), (1, ))
    assert_size_stride(arg89_1, (144, ), (1, ))
    assert_size_stride(arg90_1, (288, 144), (144, 1))
    assert_size_stride(arg91_1, (288, ), (1, ))
    assert_size_stride(arg92_1, (144, 288), (288, 1))
    assert_size_stride(arg93_1, (144, ), (1, ))
    assert_size_stride(arg94_1, (144, ), (1, ))
    assert_size_stride(arg95_1, (144, ), (1, ))
    assert_size_stride(arg96_1, (432, 144), (144, 1))
    assert_size_stride(arg97_1, (432, ), (1, ))
    assert_size_stride(arg98_1, (144, 144), (144, 1))
    assert_size_stride(arg99_1, (144, ), (1, ))
    assert_size_stride(arg100_1, (144, ), (1, ))
    assert_size_stride(arg101_1, (144, ), (1, ))
    assert_size_stride(arg102_1, (288, 144), (144, 1))
    assert_size_stride(arg103_1, (288, ), (1, ))
    assert_size_stride(arg104_1, (144, 288), (288, 1))
    assert_size_stride(arg105_1, (144, ), (1, ))
    assert_size_stride(arg106_1, (144, ), (1, ))
    assert_size_stride(arg107_1, (144, ), (1, ))
    assert_size_stride(arg108_1, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg109_1, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg110_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg111_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg113_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg114_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (576, 192), (192, 1))
    assert_size_stride(arg118_1, (576, ), (1, ))
    assert_size_stride(arg119_1, (192, 192), (192, 1))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (384, 192), (192, 1))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (192, 384), (384, 1))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (576, 192), (192, 1))
    assert_size_stride(arg130_1, (576, ), (1, ))
    assert_size_stride(arg131_1, (192, 192), (192, 1))
    assert_size_stride(arg132_1, (192, ), (1, ))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (384, 192), (192, 1))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (192, 384), (384, 1))
    assert_size_stride(arg138_1, (192, ), (1, ))
    assert_size_stride(arg139_1, (192, ), (1, ))
    assert_size_stride(arg140_1, (192, ), (1, ))
    assert_size_stride(arg141_1, (576, 192), (192, 1))
    assert_size_stride(arg142_1, (576, ), (1, ))
    assert_size_stride(arg143_1, (192, 192), (192, 1))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (384, 192), (192, 1))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (192, 384), (384, 1))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (192, ), (1, ))
    assert_size_stride(arg153_1, (576, 192), (192, 1))
    assert_size_stride(arg154_1, (576, ), (1, ))
    assert_size_stride(arg155_1, (192, 192), (192, 1))
    assert_size_stride(arg156_1, (192, ), (1, ))
    assert_size_stride(arg157_1, (192, ), (1, ))
    assert_size_stride(arg158_1, (192, ), (1, ))
    assert_size_stride(arg159_1, (384, 192), (192, 1))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (192, 384), (384, 1))
    assert_size_stride(arg162_1, (192, ), (1, ))
    assert_size_stride(arg163_1, (192, ), (1, ))
    assert_size_stride(arg164_1, (192, ), (1, ))
    assert_size_stride(arg165_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg166_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg167_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg168_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg169_1, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg170_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg171_1, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg172_1, (240, ), (1, ))
    assert_size_stride(arg173_1, (240, ), (1, ))
    assert_size_stride(arg174_1, (720, 240), (240, 1))
    assert_size_stride(arg175_1, (720, ), (1, ))
    assert_size_stride(arg176_1, (240, 240), (240, 1))
    assert_size_stride(arg177_1, (240, ), (1, ))
    assert_size_stride(arg178_1, (240, ), (1, ))
    assert_size_stride(arg179_1, (240, ), (1, ))
    assert_size_stride(arg180_1, (480, 240), (240, 1))
    assert_size_stride(arg181_1, (480, ), (1, ))
    assert_size_stride(arg182_1, (240, 480), (480, 1))
    assert_size_stride(arg183_1, (240, ), (1, ))
    assert_size_stride(arg184_1, (240, ), (1, ))
    assert_size_stride(arg185_1, (240, ), (1, ))
    assert_size_stride(arg186_1, (720, 240), (240, 1))
    assert_size_stride(arg187_1, (720, ), (1, ))
    assert_size_stride(arg188_1, (240, 240), (240, 1))
    assert_size_stride(arg189_1, (240, ), (1, ))
    assert_size_stride(arg190_1, (240, ), (1, ))
    assert_size_stride(arg191_1, (240, ), (1, ))
    assert_size_stride(arg192_1, (480, 240), (240, 1))
    assert_size_stride(arg193_1, (480, ), (1, ))
    assert_size_stride(arg194_1, (240, 480), (480, 1))
    assert_size_stride(arg195_1, (240, ), (1, ))
    assert_size_stride(arg196_1, (240, ), (1, ))
    assert_size_stride(arg197_1, (240, ), (1, ))
    assert_size_stride(arg198_1, (720, 240), (240, 1))
    assert_size_stride(arg199_1, (720, ), (1, ))
    assert_size_stride(arg200_1, (240, 240), (240, 1))
    assert_size_stride(arg201_1, (240, ), (1, ))
    assert_size_stride(arg202_1, (240, ), (1, ))
    assert_size_stride(arg203_1, (240, ), (1, ))
    assert_size_stride(arg204_1, (480, 240), (240, 1))
    assert_size_stride(arg205_1, (480, ), (1, ))
    assert_size_stride(arg206_1, (240, 480), (480, 1))
    assert_size_stride(arg207_1, (240, ), (1, ))
    assert_size_stride(arg208_1, (240, ), (1, ))
    assert_size_stride(arg209_1, (240, ), (1, ))
    assert_size_stride(arg210_1, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg211_1, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg212_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg213_1, (1000, 640), (640, 1))
    assert_size_stride(arg214_1, (1000, ), (1, ))
    assert_size_stride(arg215_1, (16, ), (1, ))
    assert_size_stride(arg216_1, (16, ), (1, ))
    assert_size_stride(arg217_1, (64, ), (1, ))
    assert_size_stride(arg218_1, (64, ), (1, ))
    assert_size_stride(arg219_1, (64, ), (1, ))
    assert_size_stride(arg220_1, (64, ), (1, ))
    assert_size_stride(arg221_1, (32, ), (1, ))
    assert_size_stride(arg222_1, (32, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (64, ), (1, ))
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (64, ), (1, ))
    assert_size_stride(arg234_1, (64, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (64, ), (1, ))
    assert_size_stride(arg240_1, (64, ), (1, ))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (96, ), (1, ))
    assert_size_stride(arg246_1, (96, ), (1, ))
    assert_size_stride(arg247_1, (96, ), (1, ))
    assert_size_stride(arg248_1, (96, ), (1, ))
    assert_size_stride(arg249_1, (96, ), (1, ))
    assert_size_stride(arg250_1, (96, ), (1, ))
    assert_size_stride(arg251_1, (96, ), (1, ))
    assert_size_stride(arg252_1, (96, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (160, ), (1, ))
    assert_size_stride(arg270_1, (160, ), (1, ))
    assert_size_stride(arg271_1, (160, ), (1, ))
    assert_size_stride(arg272_1, (160, ), (1, ))
    assert_size_stride(arg273_1, (160, ), (1, ))
    assert_size_stride(arg274_1, (160, ), (1, ))
    assert_size_stride(arg275_1, (160, ), (1, ))
    assert_size_stride(arg276_1, (160, ), (1, ))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (640, ), (1, ))
    assert_size_stride(arg279_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg279_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg279_1
    del arg64_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_1(c_void_p(buf4.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg215_1
    del arg216_1
    # Source Nodes: [shortcut, x_6], Original ATen: [aten.convolution, aten.silu]
    buf5 = extern_kernels.convolution(buf4, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del arg65_1
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_2(c_void_p(buf7.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg217_1
    del arg218_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_11, x_12], Original ATen: [aten.convolution, aten.silu]
    buf8 = extern_kernels.convolution(buf7, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf8, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del arg66_1
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = buf9; del buf9  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_3(c_void_p(buf10.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg219_1
    del arg220_1
    del arg4_1
    del arg5_1
    # Source Nodes: [x_17, x_20], Original ATen: [aten.convolution, aten.silu]
    buf11 = extern_kernels.convolution(buf10, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 32, 128, 128), (524288, 1, 4096, 32))
    del arg67_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_4(c_void_p(buf12.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_21, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf13 = extern_kernels.convolution(buf12, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    del arg68_1
    del buf12
    buf14 = buf13; del buf13  # reuse
    buf15 = buf14; del buf14  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_5(c_void_p(buf15.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg223_1
    del arg224_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.silu]
    buf16 = extern_kernels.convolution(buf15, arg69_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf16, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg69_1
    del buf15
    buf17 = buf16; del buf16  # reuse
    buf18 = buf17; del buf17  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_6(c_void_p(buf18.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg225_1
    del arg226_1
    # Source Nodes: [x_39, x_42], Original ATen: [aten.convolution, aten.silu]
    buf19 = extern_kernels.convolution(buf18, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg70_1
    del buf18
    buf20 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_7(c_void_p(buf20.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg227_1
    del arg228_1
    # Source Nodes: [x_50], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg71_1
    buf22 = buf21; del buf21  # reuse
    buf23 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_8(c_void_p(buf23.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg229_1
    del arg230_1
    # Source Nodes: [x_55, x_56], Original ATen: [aten.convolution, aten.silu]
    buf24 = extern_kernels.convolution(buf23, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf24, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg72_1
    del buf23
    buf25 = buf24; del buf24  # reuse
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_9(c_void_p(buf26.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg231_1
    del arg232_1
    # Source Nodes: [x_61, x_64], Original ATen: [aten.convolution, aten.silu]
    buf27 = extern_kernels.convolution(buf26, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg73_1
    del buf26
    buf28 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_10(c_void_p(buf28.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg233_1
    del arg234_1
    del buf27
    # Source Nodes: [x_73], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg74_1
    buf30 = buf29; del buf29  # reuse
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_11(c_void_p(buf31.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg235_1
    del arg236_1
    # Source Nodes: [x_78, x_79], Original ATen: [aten.convolution, aten.silu]
    buf32 = extern_kernels.convolution(buf31, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf32, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg75_1
    del buf31
    buf33 = buf32; del buf32  # reuse
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_12(c_void_p(buf34.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg237_1
    del arg238_1
    del arg23_1
    # Source Nodes: [x_84, x_87], Original ATen: [aten.convolution, aten.silu]
    buf35 = extern_kernels.convolution(buf34, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf35, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg76_1
    del buf34
    buf36 = buf28; del buf28  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_13(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg24_1
    del arg25_1
    del buf35
    # Source Nodes: [x_88, x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf37 = extern_kernels.convolution(buf36, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg77_1
    del buf36
    buf38 = buf37; del buf37  # reuse
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_14(c_void_p(buf39.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg241_1
    del arg242_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_101, x_102], Original ATen: [aten.convolution, aten.silu]
    buf40 = extern_kernels.convolution(buf39, arg78_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf40, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg78_1
    del buf39
    buf41 = buf40; del buf40  # reuse
    buf42 = buf41; del buf41  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_15(c_void_p(buf42.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg243_1
    del arg244_1
    del arg28_1
    del arg29_1
    # Source Nodes: [x_107, x_110], Original ATen: [aten.convolution, aten.silu]
    buf43 = extern_kernels.convolution(buf42, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (8, 96, 32, 32), (98304, 1, 3072, 96))
    del arg79_1
    del buf42
    buf44 = buf43; del buf43  # reuse
    buf45 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_16(c_void_p(buf44.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf45.data_ptr()))
    del arg245_1
    del arg246_1
    del arg30_1
    del arg31_1
    del arg80_1
    # Source Nodes: [x_118], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (8, 96, 32, 32), (98304, 1, 3072, 96))
    del buf45
    buf47 = buf46; del buf46  # reuse
    buf48 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_17(c_void_p(buf48.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg247_1
    del arg248_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_123, x_124], Original ATen: [aten.convolution, aten.silu]
    buf49 = extern_kernels.convolution(buf48, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 144, 32, 32), (147456, 1, 4608, 144))
    del arg81_1
    del buf48
    buf50 = empty_strided((32, 256, 1), (1, 32, 8192), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((32, 256, 1), (1, 32, 8192), device='cpu', dtype=torch.float32)
    buf53 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_18(c_void_p(buf49.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg82_1
    del arg83_1
    buf54 = empty((8192, 432), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf53, (8192, 144), (144, 1), 0), reinterpret_tensor(arg84_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf54)
    del arg84_1
    del arg85_1
    # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf55 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 288))
    buf56 = buf55[0]
    del buf55
    buf63 = reinterpret_tensor(buf53, (8192, 144), (144, 1), 0); del buf53  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf56, (8192, 144), (144, 1), 0), reinterpret_tensor(arg86_1, (144, 144), (1, 144), 0), alpha=1, beta=1, out=buf63)
    del arg86_1
    del arg87_1
    buf64 = reinterpret_tensor(buf51, (32, 256, 1), (256, 1, 8192), 0); del buf51  # reuse
    buf65 = reinterpret_tensor(buf50, (32, 256, 1), (256, 1, 8192), 0); del buf50  # reuse
    buf67 = reinterpret_tensor(buf56, (32, 256, 144), (36864, 144, 1), 0); del buf56  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf49.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()))
    del arg88_1
    del arg89_1
    buf68 = empty((8192, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf67, (8192, 144), (144, 1), 0), reinterpret_tensor(arg90_1, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf68)
    del arg90_1
    del arg91_1
    buf69 = reinterpret_tensor(buf68, (32, 256, 288), (73728, 288, 1), 0); del buf68  # reuse
    cpp_fused_silu_20(c_void_p(buf69.data_ptr()))
    buf70 = reinterpret_tensor(buf67, (8192, 144), (144, 1), 0); del buf67  # reuse
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf69, (8192, 288), (288, 1), 0), reinterpret_tensor(arg92_1, (288, 144), (1, 288), 0), alpha=1, beta=1, out=buf70)
    del arg92_1
    del arg93_1
    buf71 = buf65; del buf65  # reuse
    buf72 = buf64; del buf64  # reuse
    buf74 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_21(c_void_p(buf49.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg94_1
    del arg95_1
    buf75 = buf54; del buf54  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf74, (8192, 144), (144, 1), 0), reinterpret_tensor(arg96_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf75)
    del arg96_1
    del arg97_1
    # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf76 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf75, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf75, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf75, (32, 4, 256, 36), (110592, 36, 432, 1), 288))
    del buf75
    buf77 = buf76[0]
    del buf76
    buf84 = reinterpret_tensor(buf74, (8192, 144), (144, 1), 0); del buf74  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf77, (8192, 144), (144, 1), 0), reinterpret_tensor(arg98_1, (144, 144), (1, 144), 0), alpha=1, beta=1, out=buf84)
    del arg98_1
    del arg99_1
    buf85 = buf72; del buf72  # reuse
    buf86 = buf71; del buf71  # reuse
    buf88 = reinterpret_tensor(buf77, (32, 256, 144), (36864, 144, 1), 0); del buf77  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf49.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg100_1
    del arg101_1
    buf89 = reinterpret_tensor(buf69, (8192, 288), (288, 1), 0); del buf69  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf88, (8192, 144), (144, 1), 0), reinterpret_tensor(arg102_1, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf89)
    del arg102_1
    del arg103_1
    buf90 = reinterpret_tensor(buf89, (32, 256, 288), (73728, 288, 1), 0); del buf89  # reuse
    cpp_fused_silu_23(c_void_p(buf90.data_ptr()))
    buf91 = reinterpret_tensor(buf88, (8192, 144), (144, 1), 0); del buf88  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf90, (8192, 288), (288, 1), 0), reinterpret_tensor(arg104_1, (288, 144), (1, 288), 0), alpha=1, beta=1, out=buf91)
    del arg104_1
    del arg105_1
    del buf90
    buf92 = reinterpret_tensor(buf91, (32, 256, 144), (36864, 144, 1), 0); del buf91  # reuse
    buf93 = buf86; del buf86  # reuse
    buf94 = buf85; del buf85  # reuse
    buf96 = empty((18432, 2, 16, 2), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 144, 32, 32), (147456, 1, 4608, 144), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_convolution_native_layer_norm_24(c_void_p(buf92.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg106_1
    del arg107_1
    del buf49
    del buf63
    del buf70
    del buf84
    del buf92
    del buf93
    del buf94
    del buf96
    # Source Nodes: [x_156], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (8, 96, 32, 32), (98304, 1, 3072, 96))
    del arg108_1
    buf99 = buf98; del buf98  # reuse
    buf100 = reinterpret_tensor(buf0, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf0  # reuse
    buf101 = empty_strided((96, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_25(c_void_p(buf99.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg109_1
    del arg249_1
    del arg250_1
    del arg34_1
    del arg35_1
    del buf44
    del buf99
    # Source Nodes: [cat_5, x_162], Original ATen: [aten.cat, aten.convolution]
    buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 96, 32, 32), (98304, 1, 3072, 96))
    del buf100
    del buf101
    buf103 = buf102; del buf102  # reuse
    buf104 = buf103; del buf103  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_26(c_void_p(buf104.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg36_1
    del arg37_1
    # Source Nodes: [shortcut_6, x_168], Original ATen: [aten.convolution, aten.silu]
    buf105 = extern_kernels.convolution(buf104, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf105, (8, 384, 32, 32), (393216, 1, 12288, 384))
    del arg110_1
    del buf104
    buf106 = buf105; del buf105  # reuse
    buf107 = buf106; del buf106  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_27(c_void_p(buf107.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg253_1
    del arg254_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_173, x_174], Original ATen: [aten.convolution, aten.silu]
    buf108 = extern_kernels.convolution(buf107, arg111_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf108, (8, 384, 16, 16), (98304, 1, 6144, 384))
    del arg111_1
    del buf107
    buf109 = buf108; del buf108  # reuse
    buf110 = buf109; del buf109  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_28(c_void_p(buf110.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg255_1
    del arg256_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_179, x_182], Original ATen: [aten.convolution, aten.silu]
    buf111 = extern_kernels.convolution(buf110, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf111, (8, 128, 16, 16), (32768, 1, 2048, 128))
    del arg112_1
    buf112 = buf111; del buf111  # reuse
    buf113 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_29(c_void_p(buf112.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf113.data_ptr()))
    del arg113_1
    del arg257_1
    del arg258_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_190], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf112, buf113, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 128, 16, 16), (32768, 1, 2048, 128))
    del buf113
    buf115 = buf114; del buf114  # reuse
    buf116 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_30(c_void_p(buf116.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg259_1
    del arg260_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_195, x_196], Original ATen: [aten.convolution, aten.silu]
    buf117 = extern_kernels.convolution(buf116, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf117, (8, 192, 16, 16), (49152, 1, 3072, 192))
    del arg114_1
    del buf116
    buf118 = empty_strided((32, 64, 1), (1, 32, 2048), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((32, 64, 1), (1, 32, 2048), device='cpu', dtype=torch.float32)
    buf121 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_31(c_void_p(buf117.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg115_1
    del arg116_1
    buf122 = reinterpret_tensor(buf97, (2048, 576), (576, 1), 0); del buf97  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf121, (2048, 192), (192, 1), 0), reinterpret_tensor(arg117_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf122)
    del arg117_1
    del arg118_1
    # Source Nodes: [x_199], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf123 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf122, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf122, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf122, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf124 = buf123[0]
    del buf123
    buf131 = reinterpret_tensor(buf121, (2048, 192), (192, 1), 0); del buf121  # reuse
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf124, (2048, 192), (192, 1), 0), reinterpret_tensor(arg119_1, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf131)
    del arg119_1
    del arg120_1
    buf132 = reinterpret_tensor(buf119, (32, 64, 1), (64, 1, 2048), 0); del buf119  # reuse
    buf133 = reinterpret_tensor(buf118, (32, 64, 1), (64, 1, 2048), 0); del buf118  # reuse
    buf135 = reinterpret_tensor(buf124, (32, 64, 192), (12288, 192, 1), 0); del buf124  # reuse
    cpp_fused_add_native_layer_norm_32(c_void_p(buf117.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg121_1
    del arg122_1
    buf136 = reinterpret_tensor(buf110, (2048, 384), (384, 1), 0); del buf110  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf135, (2048, 192), (192, 1), 0), reinterpret_tensor(arg123_1, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf136)
    del arg123_1
    del arg124_1
    buf137 = reinterpret_tensor(buf136, (32, 64, 384), (24576, 384, 1), 0); del buf136  # reuse
    cpp_fused_silu_33(c_void_p(buf137.data_ptr()))
    buf138 = reinterpret_tensor(buf135, (2048, 192), (192, 1), 0); del buf135  # reuse
    # Source Nodes: [x_208], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf137, (2048, 384), (384, 1), 0), reinterpret_tensor(arg125_1, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf138)
    del arg125_1
    del arg126_1
    buf139 = buf133; del buf133  # reuse
    buf140 = buf132; del buf132  # reuse
    buf142 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_34(c_void_p(buf117.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    del arg127_1
    del arg128_1
    buf143 = buf122; del buf122  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf142, (2048, 192), (192, 1), 0), reinterpret_tensor(arg129_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf143)
    del arg129_1
    del arg130_1
    # Source Nodes: [x_211], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf144 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf143, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf143, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf143, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf145 = buf144[0]
    del buf144
    buf152 = reinterpret_tensor(buf142, (2048, 192), (192, 1), 0); del buf142  # reuse
    # Source Nodes: [x_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf145, (2048, 192), (192, 1), 0), reinterpret_tensor(arg131_1, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf152)
    del arg131_1
    del arg132_1
    buf153 = buf140; del buf140  # reuse
    buf154 = buf139; del buf139  # reuse
    buf156 = reinterpret_tensor(buf145, (32, 64, 192), (12288, 192, 1), 0); del buf145  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf117.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg133_1
    del arg134_1
    buf157 = reinterpret_tensor(buf137, (2048, 384), (384, 1), 0); del buf137  # reuse
    # Source Nodes: [x_216], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf156, (2048, 192), (192, 1), 0), reinterpret_tensor(arg135_1, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf157)
    del arg135_1
    del arg136_1
    buf158 = reinterpret_tensor(buf157, (32, 64, 384), (24576, 384, 1), 0); del buf157  # reuse
    cpp_fused_silu_36(c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf156, (2048, 192), (192, 1), 0); del buf156  # reuse
    # Source Nodes: [x_220], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf158, (2048, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf159)
    del arg137_1
    del arg138_1
    buf160 = reinterpret_tensor(buf159, (32, 64, 192), (12288, 192, 1), 0); del buf159  # reuse
    buf161 = buf154; del buf154  # reuse
    buf162 = buf153; del buf153  # reuse
    buf164 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_37(c_void_p(buf160.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()))
    del arg139_1
    del arg140_1
    del buf117
    buf165 = buf143; del buf143  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf164, (2048, 192), (192, 1), 0), reinterpret_tensor(arg141_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf165)
    del arg141_1
    del arg142_1
    # Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf166 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf165, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf165, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf165, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf167 = buf166[0]
    del buf166
    buf174 = reinterpret_tensor(buf164, (2048, 192), (192, 1), 0); del buf164  # reuse
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf167, (2048, 192), (192, 1), 0), reinterpret_tensor(arg143_1, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf174)
    del arg143_1
    del arg144_1
    buf175 = buf162; del buf162  # reuse
    buf176 = buf161; del buf161  # reuse
    buf178 = reinterpret_tensor(buf167, (32, 64, 192), (12288, 192, 1), 0); del buf167  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf160.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg145_1
    del arg146_1
    buf179 = reinterpret_tensor(buf158, (2048, 384), (384, 1), 0); del buf158  # reuse
    # Source Nodes: [x_228], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf178, (2048, 192), (192, 1), 0), reinterpret_tensor(arg147_1, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf179)
    del arg147_1
    del arg148_1
    buf180 = reinterpret_tensor(buf179, (32, 64, 384), (24576, 384, 1), 0); del buf179  # reuse
    cpp_fused_silu_39(c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf178, (2048, 192), (192, 1), 0); del buf178  # reuse
    # Source Nodes: [x_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf180, (2048, 384), (384, 1), 0), reinterpret_tensor(arg149_1, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf181)
    del arg149_1
    del arg150_1
    buf182 = buf176; del buf176  # reuse
    buf183 = buf175; del buf175  # reuse
    buf185 = reinterpret_tensor(buf152, (32, 64, 192), (12288, 192, 1), 0); del buf152  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf160.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()))
    del arg151_1
    del arg152_1
    buf186 = buf165; del buf165  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf185, (2048, 192), (192, 1), 0), reinterpret_tensor(arg153_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf186)
    del arg153_1
    del arg154_1
    # Source Nodes: [x_235], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf187 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf186, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf186, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf186, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    del buf186
    buf188 = buf187[0]
    del buf187
    buf195 = reinterpret_tensor(buf185, (2048, 192), (192, 1), 0); del buf185  # reuse
    # Source Nodes: [x_237], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf188, (2048, 192), (192, 1), 0), reinterpret_tensor(arg155_1, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf195)
    del arg155_1
    del arg156_1
    buf196 = buf183; del buf183  # reuse
    buf197 = buf182; del buf182  # reuse
    buf199 = reinterpret_tensor(buf188, (32, 64, 192), (12288, 192, 1), 0); del buf188  # reuse
    cpp_fused_add_native_layer_norm_41(c_void_p(buf160.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg157_1
    del arg158_1
    buf200 = reinterpret_tensor(buf180, (2048, 384), (384, 1), 0); del buf180  # reuse
    # Source Nodes: [x_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf199, (2048, 192), (192, 1), 0), reinterpret_tensor(arg159_1, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf200)
    del arg159_1
    del arg160_1
    buf201 = reinterpret_tensor(buf200, (32, 64, 384), (24576, 384, 1), 0); del buf200  # reuse
    cpp_fused_silu_42(c_void_p(buf201.data_ptr()))
    buf202 = reinterpret_tensor(buf199, (2048, 192), (192, 1), 0); del buf199  # reuse
    # Source Nodes: [x_244], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf201, (2048, 384), (384, 1), 0), reinterpret_tensor(arg161_1, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf202)
    del arg161_1
    del arg162_1
    del buf201
    buf203 = reinterpret_tensor(buf202, (32, 64, 192), (12288, 192, 1), 0); del buf202  # reuse
    buf204 = buf197; del buf197  # reuse
    buf205 = buf196; del buf196  # reuse
    buf207 = reinterpret_tensor(buf138, (12288, 2, 8, 2), (32, 16, 2, 1), 0); del buf138  # reuse
    buf208 = reinterpret_tensor(buf131, (8, 192, 16, 16), (49152, 1, 3072, 192), 0); del buf131  # reuse
    cpp_fused_add_clone_convolution_native_layer_norm_43(c_void_p(buf203.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg163_1
    del arg164_1
    del buf160
    del buf174
    del buf181
    del buf195
    del buf203
    del buf204
    del buf205
    del buf207
    # Source Nodes: [x_252], Original ATen: [aten.convolution]
    buf209 = extern_kernels.convolution(buf208, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf209, (8, 128, 16, 16), (32768, 1, 2048, 128))
    del arg165_1
    del buf208
    buf210 = buf209; del buf209  # reuse
    buf211 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    buf212 = empty_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_44(c_void_p(buf210.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg166_1
    del arg261_1
    del arg262_1
    del arg46_1
    del arg47_1
    del buf112
    del buf210
    # Source Nodes: [cat_4, x_258], Original ATen: [aten.cat, aten.convolution]
    buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf213, (8, 128, 16, 16), (32768, 1, 2048, 128))
    del buf211
    del buf212
    buf214 = buf213; del buf213  # reuse
    buf215 = buf214; del buf214  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_45(c_void_p(buf215.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg263_1
    del arg264_1
    del arg48_1
    del arg49_1
    # Source Nodes: [shortcut_8, x_264], Original ATen: [aten.convolution, aten.silu]
    buf216 = extern_kernels.convolution(buf215, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf216, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del arg167_1
    del buf215
    buf217 = buf216; del buf216  # reuse
    buf218 = buf217; del buf217  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_46(c_void_p(buf218.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg265_1
    del arg266_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_269, x_270], Original ATen: [aten.convolution, aten.silu]
    buf219 = extern_kernels.convolution(buf218, arg168_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
    assert_size_stride(buf219, (8, 512, 8, 8), (32768, 1, 4096, 512))
    del arg168_1
    del buf218
    buf220 = buf219; del buf219  # reuse
    buf221 = buf220; del buf220  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_47(c_void_p(buf221.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg267_1
    del arg268_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_275, x_278], Original ATen: [aten.convolution, aten.silu]
    buf222 = extern_kernels.convolution(buf221, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf222, (8, 160, 8, 8), (10240, 1, 1280, 160))
    del arg169_1
    del buf221
    buf223 = buf222; del buf222  # reuse
    buf224 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_48(c_void_p(buf223.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf224.data_ptr()))
    del arg170_1
    del arg269_1
    del arg270_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_286], Original ATen: [aten.convolution]
    buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf225, (8, 160, 8, 8), (10240, 1, 1280, 160))
    del buf224
    buf226 = buf225; del buf225  # reuse
    buf227 = buf226; del buf226  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_49(c_void_p(buf227.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_291, x_292], Original ATen: [aten.convolution, aten.silu]
    buf228 = extern_kernels.convolution(buf227, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (8, 240, 8, 8), (15360, 1, 1920, 240))
    del arg171_1
    del buf227
    buf229 = empty_strided((32, 16, 1), (1, 32, 512), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((32, 16, 1), (1, 32, 512), device='cpu', dtype=torch.float32)
    buf232 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_50(c_void_p(buf228.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg172_1
    del arg173_1
    buf233 = empty((512, 720), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf232, (512, 240), (240, 1), 0), reinterpret_tensor(arg174_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf233)
    del arg174_1
    del arg175_1
    # Source Nodes: [x_295], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf234 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf233, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf233, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf233, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    buf235 = buf234[0]
    del buf234
    buf242 = reinterpret_tensor(buf232, (512, 240), (240, 1), 0); del buf232  # reuse
    # Source Nodes: [x_297], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf235, (512, 240), (240, 1), 0), reinterpret_tensor(arg176_1, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf242)
    del arg176_1
    del arg177_1
    buf243 = reinterpret_tensor(buf230, (32, 16, 1), (16, 1, 512), 0); del buf230  # reuse
    buf244 = reinterpret_tensor(buf229, (32, 16, 1), (16, 1, 512), 0); del buf229  # reuse
    buf246 = reinterpret_tensor(buf235, (32, 16, 240), (3840, 240, 1), 0); del buf235  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf228.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()))
    del arg178_1
    del arg179_1
    buf247 = empty((512, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf246, (512, 240), (240, 1), 0), reinterpret_tensor(arg180_1, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf247)
    del arg180_1
    del arg181_1
    buf248 = reinterpret_tensor(buf247, (32, 16, 480), (7680, 480, 1), 0); del buf247  # reuse
    cpp_fused_silu_52(c_void_p(buf248.data_ptr()))
    buf249 = reinterpret_tensor(buf246, (512, 240), (240, 1), 0); del buf246  # reuse
    # Source Nodes: [x_304], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf248, (512, 480), (480, 1), 0), reinterpret_tensor(arg182_1, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf249)
    del arg182_1
    del arg183_1
    buf250 = buf244; del buf244  # reuse
    buf251 = buf243; del buf243  # reuse
    buf253 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_53(c_void_p(buf228.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    del arg184_1
    del arg185_1
    buf254 = buf233; del buf233  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf253, (512, 240), (240, 1), 0), reinterpret_tensor(arg186_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf254)
    del arg186_1
    del arg187_1
    # Source Nodes: [x_307], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf255 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf254, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf254, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf254, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    buf256 = buf255[0]
    del buf255
    buf263 = reinterpret_tensor(buf253, (512, 240), (240, 1), 0); del buf253  # reuse
    # Source Nodes: [x_309], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg189_1, reinterpret_tensor(buf256, (512, 240), (240, 1), 0), reinterpret_tensor(arg188_1, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf263)
    del arg188_1
    del arg189_1
    buf264 = buf251; del buf251  # reuse
    buf265 = buf250; del buf250  # reuse
    buf267 = reinterpret_tensor(buf256, (32, 16, 240), (3840, 240, 1), 0); del buf256  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf228.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()))
    del arg190_1
    del arg191_1
    buf268 = reinterpret_tensor(buf248, (512, 480), (480, 1), 0); del buf248  # reuse
    # Source Nodes: [x_312], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf267, (512, 240), (240, 1), 0), reinterpret_tensor(arg192_1, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf268)
    del arg192_1
    del arg193_1
    buf269 = reinterpret_tensor(buf268, (32, 16, 480), (7680, 480, 1), 0); del buf268  # reuse
    cpp_fused_silu_55(c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf267, (512, 240), (240, 1), 0); del buf267  # reuse
    # Source Nodes: [x_316], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg195_1, reinterpret_tensor(buf269, (512, 480), (480, 1), 0), reinterpret_tensor(arg194_1, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf270)
    del arg194_1
    del arg195_1
    buf271 = reinterpret_tensor(buf270, (32, 16, 240), (3840, 240, 1), 0); del buf270  # reuse
    buf272 = buf265; del buf265  # reuse
    buf273 = buf264; del buf264  # reuse
    buf275 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_56(c_void_p(buf271.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()))
    del arg196_1
    del arg197_1
    del buf228
    del buf242
    buf276 = buf254; del buf254  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf275, (512, 240), (240, 1), 0), reinterpret_tensor(arg198_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf276)
    del arg198_1
    del arg199_1
    # Source Nodes: [x_319], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf277 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf276, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf276, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf276, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    del buf276
    buf278 = buf277[0]
    del buf277
    buf285 = reinterpret_tensor(buf275, (512, 240), (240, 1), 0); del buf275  # reuse
    # Source Nodes: [x_321], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg201_1, reinterpret_tensor(buf278, (512, 240), (240, 1), 0), reinterpret_tensor(arg200_1, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf285)
    del arg200_1
    del arg201_1
    buf286 = buf273; del buf273  # reuse
    buf287 = buf272; del buf272  # reuse
    buf289 = reinterpret_tensor(buf278, (32, 16, 240), (3840, 240, 1), 0); del buf278  # reuse
    cpp_fused_add_native_layer_norm_57(c_void_p(buf271.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    del arg202_1
    del arg203_1
    buf290 = reinterpret_tensor(buf269, (512, 480), (480, 1), 0); del buf269  # reuse
    # Source Nodes: [x_324], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf289, (512, 240), (240, 1), 0), reinterpret_tensor(arg204_1, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf290)
    del arg204_1
    del arg205_1
    buf291 = reinterpret_tensor(buf290, (32, 16, 480), (7680, 480, 1), 0); del buf290  # reuse
    cpp_fused_silu_58(c_void_p(buf291.data_ptr()))
    buf292 = reinterpret_tensor(buf289, (512, 240), (240, 1), 0); del buf289  # reuse
    # Source Nodes: [x_328], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf291, (512, 480), (480, 1), 0), reinterpret_tensor(arg206_1, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf292)
    del arg206_1
    del arg207_1
    del buf291
    buf293 = buf287; del buf287  # reuse
    buf294 = buf286; del buf286  # reuse
    buf296 = reinterpret_tensor(buf263, (7680, 2, 4, 2), (16, 8, 2, 1), 0); del buf263  # reuse
    buf297 = reinterpret_tensor(buf249, (8, 240, 8, 8), (15360, 1, 1920, 240), 0); del buf249  # reuse
    cpp_fused_add_clone_convolution_native_layer_norm_59(c_void_p(buf271.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg208_1
    del arg209_1
    del buf271
    del buf285
    del buf292
    del buf293
    del buf294
    del buf296
    # Source Nodes: [x_336], Original ATen: [aten.convolution]
    buf298 = extern_kernels.convolution(buf297, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf298, (8, 160, 8, 8), (10240, 1, 1280, 160))
    del arg210_1
    del buf297
    buf299 = buf298; del buf298  # reuse
    buf300 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    buf301 = empty_strided((160, 320, 3, 3), (2880, 1, 960, 320), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_60(c_void_p(buf299.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del arg211_1
    del arg273_1
    del arg274_1
    del arg58_1
    del arg59_1
    del buf223
    del buf299
    # Source Nodes: [cat_3, x_342], Original ATen: [aten.cat, aten.convolution]
    buf302 = extern_kernels.convolution(buf300, buf301, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf302, (8, 160, 8, 8), (10240, 1, 1280, 160))
    del buf300
    del buf301
    buf303 = buf302; del buf302  # reuse
    buf304 = buf303; del buf303  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_61(c_void_p(buf304.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg275_1
    del arg276_1
    del arg60_1
    del arg61_1
    # Source Nodes: [x_348, x_349], Original ATen: [aten.convolution, aten.silu]
    buf305 = extern_kernels.convolution(buf304, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf305, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg212_1
    del buf304
    buf306 = buf305; del buf305  # reuse
    buf307 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf307, (8, 640, 1, 1), (640, 1, 1, 1), 0); del buf307  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_62(c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg62_1
    del arg63_1
    del buf306
    buf309 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_360], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf308, (8, 640), (640, 1), 0), reinterpret_tensor(arg213_1, (640, 1000), (1, 640), 0), alpha=1, beta=1, out=buf309)
    del arg213_1
    del arg214_1
    return (buf309, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1000, 640), (640, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
