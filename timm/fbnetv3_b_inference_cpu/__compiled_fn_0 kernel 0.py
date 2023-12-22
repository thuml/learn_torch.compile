
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


cpp_fused__native_batch_norm_legit_no_training_add_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
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


cpp_fused__native_batch_norm_legit_no_training_add_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_17 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3932160L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (122880L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(983040L); x0+=static_cast<long>(8L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (122880L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(983040L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (122880L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_33 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(983040L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (122880L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(983040L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x2) + (122880L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (122880L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1638400L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_44 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(409600L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_45 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_46 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_47 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_49 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_56 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_add_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_61 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_62 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_71 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_76 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_81 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_83 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x2) + (92160L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (92160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_88 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1474560L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_89 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (720L*x2) + (46080L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5760L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_91 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(720L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (720L*x1) + (46080L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (720L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (720L*x1) + (46080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_92 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_93 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376832L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_94 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x2) + (47104L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_96 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_98 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376832L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_99 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x2) + (47104L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_101 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_103 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376832L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_104 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x2) + (47104L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_106 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_108 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376832L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_109 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x2) + (47104L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_111 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_113 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376832L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_114 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x2) + (47104L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_116 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (47104L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_118 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(565248L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_119 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x2) + (70656L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8832L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_hardsigmoid_hardswish_mul_121 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1104L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1104L*x1) + (70656L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1104L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (1104L*x1) + (70656L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_122 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_123 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1344L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1344L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1344L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1344L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1344L*x2) + (86016L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1344L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(10752L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(15872L); x0+=static_cast<long>(8L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (24, ), (1, ))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (48, ), (1, ))
    assert_size_stride(arg17_1, (48, ), (1, ))
    assert_size_stride(arg18_1, (48, ), (1, ))
    assert_size_stride(arg19_1, (48, ), (1, ))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (48, ), (1, ))
    assert_size_stride(arg23_1, (48, ), (1, ))
    assert_size_stride(arg24_1, (48, ), (1, ))
    assert_size_stride(arg25_1, (48, ), (1, ))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (48, ), (1, ))
    assert_size_stride(arg29_1, (48, ), (1, ))
    assert_size_stride(arg30_1, (48, ), (1, ))
    assert_size_stride(arg31_1, (48, ), (1, ))
    assert_size_stride(arg32_1, (24, ), (1, ))
    assert_size_stride(arg33_1, (24, ), (1, ))
    assert_size_stride(arg34_1, (120, ), (1, ))
    assert_size_stride(arg35_1, (120, ), (1, ))
    assert_size_stride(arg36_1, (120, ), (1, ))
    assert_size_stride(arg37_1, (120, ), (1, ))
    assert_size_stride(arg38_1, (40, ), (1, ))
    assert_size_stride(arg39_1, (40, ), (1, ))
    assert_size_stride(arg40_1, (120, ), (1, ))
    assert_size_stride(arg41_1, (120, ), (1, ))
    assert_size_stride(arg42_1, (120, ), (1, ))
    assert_size_stride(arg43_1, (120, ), (1, ))
    assert_size_stride(arg44_1, (40, ), (1, ))
    assert_size_stride(arg45_1, (40, ), (1, ))
    assert_size_stride(arg46_1, (120, ), (1, ))
    assert_size_stride(arg47_1, (120, ), (1, ))
    assert_size_stride(arg48_1, (120, ), (1, ))
    assert_size_stride(arg49_1, (120, ), (1, ))
    assert_size_stride(arg50_1, (40, ), (1, ))
    assert_size_stride(arg51_1, (40, ), (1, ))
    assert_size_stride(arg52_1, (120, ), (1, ))
    assert_size_stride(arg53_1, (120, ), (1, ))
    assert_size_stride(arg54_1, (120, ), (1, ))
    assert_size_stride(arg55_1, (120, ), (1, ))
    assert_size_stride(arg56_1, (40, ), (1, ))
    assert_size_stride(arg57_1, (40, ), (1, ))
    assert_size_stride(arg58_1, (120, ), (1, ))
    assert_size_stride(arg59_1, (120, ), (1, ))
    assert_size_stride(arg60_1, (120, ), (1, ))
    assert_size_stride(arg61_1, (120, ), (1, ))
    assert_size_stride(arg62_1, (40, ), (1, ))
    assert_size_stride(arg63_1, (40, ), (1, ))
    assert_size_stride(arg64_1, (200, ), (1, ))
    assert_size_stride(arg65_1, (200, ), (1, ))
    assert_size_stride(arg66_1, (200, ), (1, ))
    assert_size_stride(arg67_1, (200, ), (1, ))
    assert_size_stride(arg68_1, (72, ), (1, ))
    assert_size_stride(arg69_1, (72, ), (1, ))
    assert_size_stride(arg70_1, (216, ), (1, ))
    assert_size_stride(arg71_1, (216, ), (1, ))
    assert_size_stride(arg72_1, (216, ), (1, ))
    assert_size_stride(arg73_1, (216, ), (1, ))
    assert_size_stride(arg74_1, (72, ), (1, ))
    assert_size_stride(arg75_1, (72, ), (1, ))
    assert_size_stride(arg76_1, (216, ), (1, ))
    assert_size_stride(arg77_1, (216, ), (1, ))
    assert_size_stride(arg78_1, (216, ), (1, ))
    assert_size_stride(arg79_1, (216, ), (1, ))
    assert_size_stride(arg80_1, (72, ), (1, ))
    assert_size_stride(arg81_1, (72, ), (1, ))
    assert_size_stride(arg82_1, (216, ), (1, ))
    assert_size_stride(arg83_1, (216, ), (1, ))
    assert_size_stride(arg84_1, (216, ), (1, ))
    assert_size_stride(arg85_1, (216, ), (1, ))
    assert_size_stride(arg86_1, (72, ), (1, ))
    assert_size_stride(arg87_1, (72, ), (1, ))
    assert_size_stride(arg88_1, (216, ), (1, ))
    assert_size_stride(arg89_1, (216, ), (1, ))
    assert_size_stride(arg90_1, (216, ), (1, ))
    assert_size_stride(arg91_1, (216, ), (1, ))
    assert_size_stride(arg92_1, (72, ), (1, ))
    assert_size_stride(arg93_1, (72, ), (1, ))
    assert_size_stride(arg94_1, (360, ), (1, ))
    assert_size_stride(arg95_1, (360, ), (1, ))
    assert_size_stride(arg96_1, (360, ), (1, ))
    assert_size_stride(arg97_1, (360, ), (1, ))
    assert_size_stride(arg98_1, (120, ), (1, ))
    assert_size_stride(arg99_1, (120, ), (1, ))
    assert_size_stride(arg100_1, (360, ), (1, ))
    assert_size_stride(arg101_1, (360, ), (1, ))
    assert_size_stride(arg102_1, (360, ), (1, ))
    assert_size_stride(arg103_1, (360, ), (1, ))
    assert_size_stride(arg104_1, (120, ), (1, ))
    assert_size_stride(arg105_1, (120, ), (1, ))
    assert_size_stride(arg106_1, (360, ), (1, ))
    assert_size_stride(arg107_1, (360, ), (1, ))
    assert_size_stride(arg108_1, (360, ), (1, ))
    assert_size_stride(arg109_1, (360, ), (1, ))
    assert_size_stride(arg110_1, (120, ), (1, ))
    assert_size_stride(arg111_1, (120, ), (1, ))
    assert_size_stride(arg112_1, (360, ), (1, ))
    assert_size_stride(arg113_1, (360, ), (1, ))
    assert_size_stride(arg114_1, (360, ), (1, ))
    assert_size_stride(arg115_1, (360, ), (1, ))
    assert_size_stride(arg116_1, (120, ), (1, ))
    assert_size_stride(arg117_1, (120, ), (1, ))
    assert_size_stride(arg118_1, (360, ), (1, ))
    assert_size_stride(arg119_1, (360, ), (1, ))
    assert_size_stride(arg120_1, (360, ), (1, ))
    assert_size_stride(arg121_1, (360, ), (1, ))
    assert_size_stride(arg122_1, (120, ), (1, ))
    assert_size_stride(arg123_1, (120, ), (1, ))
    assert_size_stride(arg124_1, (360, ), (1, ))
    assert_size_stride(arg125_1, (360, ), (1, ))
    assert_size_stride(arg126_1, (360, ), (1, ))
    assert_size_stride(arg127_1, (360, ), (1, ))
    assert_size_stride(arg128_1, (120, ), (1, ))
    assert_size_stride(arg129_1, (120, ), (1, ))
    assert_size_stride(arg130_1, (720, ), (1, ))
    assert_size_stride(arg131_1, (720, ), (1, ))
    assert_size_stride(arg132_1, (720, ), (1, ))
    assert_size_stride(arg133_1, (720, ), (1, ))
    assert_size_stride(arg134_1, (184, ), (1, ))
    assert_size_stride(arg135_1, (184, ), (1, ))
    assert_size_stride(arg136_1, (736, ), (1, ))
    assert_size_stride(arg137_1, (736, ), (1, ))
    assert_size_stride(arg138_1, (736, ), (1, ))
    assert_size_stride(arg139_1, (736, ), (1, ))
    assert_size_stride(arg140_1, (184, ), (1, ))
    assert_size_stride(arg141_1, (184, ), (1, ))
    assert_size_stride(arg142_1, (736, ), (1, ))
    assert_size_stride(arg143_1, (736, ), (1, ))
    assert_size_stride(arg144_1, (736, ), (1, ))
    assert_size_stride(arg145_1, (736, ), (1, ))
    assert_size_stride(arg146_1, (184, ), (1, ))
    assert_size_stride(arg147_1, (184, ), (1, ))
    assert_size_stride(arg148_1, (736, ), (1, ))
    assert_size_stride(arg149_1, (736, ), (1, ))
    assert_size_stride(arg150_1, (736, ), (1, ))
    assert_size_stride(arg151_1, (736, ), (1, ))
    assert_size_stride(arg152_1, (184, ), (1, ))
    assert_size_stride(arg153_1, (184, ), (1, ))
    assert_size_stride(arg154_1, (736, ), (1, ))
    assert_size_stride(arg155_1, (736, ), (1, ))
    assert_size_stride(arg156_1, (736, ), (1, ))
    assert_size_stride(arg157_1, (736, ), (1, ))
    assert_size_stride(arg158_1, (184, ), (1, ))
    assert_size_stride(arg159_1, (184, ), (1, ))
    assert_size_stride(arg160_1, (736, ), (1, ))
    assert_size_stride(arg161_1, (736, ), (1, ))
    assert_size_stride(arg162_1, (736, ), (1, ))
    assert_size_stride(arg163_1, (736, ), (1, ))
    assert_size_stride(arg164_1, (184, ), (1, ))
    assert_size_stride(arg165_1, (184, ), (1, ))
    assert_size_stride(arg166_1, (1104, ), (1, ))
    assert_size_stride(arg167_1, (1104, ), (1, ))
    assert_size_stride(arg168_1, (1104, ), (1, ))
    assert_size_stride(arg169_1, (1104, ), (1, ))
    assert_size_stride(arg170_1, (224, ), (1, ))
    assert_size_stride(arg171_1, (224, ), (1, ))
    assert_size_stride(arg172_1, (1344, ), (1, ))
    assert_size_stride(arg173_1, (1344, ), (1, ))
    assert_size_stride(arg174_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg175_1, (1000, ), (1, ))
    assert_size_stride(arg176_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg177_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg178_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg179_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg180_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg181_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg182_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg183_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg184_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg185_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg186_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg187_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg188_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg189_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg190_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg191_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg192_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg193_1, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg194_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg196_1, (8, ), (1, ))
    assert_size_stride(arg197_1, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg198_1, (120, ), (1, ))
    assert_size_stride(arg199_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg200_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg201_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg202_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg203_1, (16, ), (1, ))
    assert_size_stride(arg204_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg205_1, (120, ), (1, ))
    assert_size_stride(arg206_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg207_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg208_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg209_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg210_1, (16, ), (1, ))
    assert_size_stride(arg211_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg212_1, (120, ), (1, ))
    assert_size_stride(arg213_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg214_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg215_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg216_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg217_1, (16, ), (1, ))
    assert_size_stride(arg218_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg219_1, (120, ), (1, ))
    assert_size_stride(arg220_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg221_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg222_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg223_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg224_1, (16, ), (1, ))
    assert_size_stride(arg225_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg226_1, (120, ), (1, ))
    assert_size_stride(arg227_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg228_1, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg229_1, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg230_1, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg231_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg232_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg233_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg234_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg235_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg236_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg237_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg238_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg239_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg240_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg241_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg242_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg243_1, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg244_1, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg246_1, (24, ), (1, ))
    assert_size_stride(arg247_1, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg248_1, (360, ), (1, ))
    assert_size_stride(arg249_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg250_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg251_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg252_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg253_1, (32, ), (1, ))
    assert_size_stride(arg254_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg255_1, (360, ), (1, ))
    assert_size_stride(arg256_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg257_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg258_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg259_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg260_1, (32, ), (1, ))
    assert_size_stride(arg261_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg262_1, (360, ), (1, ))
    assert_size_stride(arg263_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg264_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg265_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg266_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg267_1, (32, ), (1, ))
    assert_size_stride(arg268_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg269_1, (360, ), (1, ))
    assert_size_stride(arg270_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg271_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg272_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg273_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg274_1, (32, ), (1, ))
    assert_size_stride(arg275_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg276_1, (360, ), (1, ))
    assert_size_stride(arg277_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg278_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg279_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg280_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg281_1, (32, ), (1, ))
    assert_size_stride(arg282_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg283_1, (360, ), (1, ))
    assert_size_stride(arg284_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg285_1, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg286_1, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg287_1, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg288_1, (32, ), (1, ))
    assert_size_stride(arg289_1, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg290_1, (720, ), (1, ))
    assert_size_stride(arg291_1, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg292_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg293_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg294_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg295_1, (48, ), (1, ))
    assert_size_stride(arg296_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg297_1, (736, ), (1, ))
    assert_size_stride(arg298_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg299_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg300_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg301_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg302_1, (48, ), (1, ))
    assert_size_stride(arg303_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg304_1, (736, ), (1, ))
    assert_size_stride(arg305_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg306_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg307_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg308_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg309_1, (48, ), (1, ))
    assert_size_stride(arg310_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg311_1, (736, ), (1, ))
    assert_size_stride(arg312_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg313_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg314_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg315_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg316_1, (48, ), (1, ))
    assert_size_stride(arg317_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg318_1, (736, ), (1, ))
    assert_size_stride(arg319_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg320_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg321_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg322_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg323_1, (48, ), (1, ))
    assert_size_stride(arg324_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg325_1, (736, ), (1, ))
    assert_size_stride(arg326_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg327_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg328_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg329_1, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg330_1, (48, ), (1, ))
    assert_size_stride(arg331_1, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg332_1, (1104, ), (1, ))
    assert_size_stride(arg333_1, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg334_1, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg335_1, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg336_1, (16, ), (1, ))
    assert_size_stride(arg337_1, (16, ), (1, ))
    assert_size_stride(arg338_1, (16, ), (1, ))
    assert_size_stride(arg339_1, (16, ), (1, ))
    assert_size_stride(arg340_1, (16, ), (1, ))
    assert_size_stride(arg341_1, (16, ), (1, ))
    assert_size_stride(arg342_1, (16, ), (1, ))
    assert_size_stride(arg343_1, (16, ), (1, ))
    assert_size_stride(arg344_1, (16, ), (1, ))
    assert_size_stride(arg345_1, (16, ), (1, ))
    assert_size_stride(arg346_1, (64, ), (1, ))
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (64, ), (1, ))
    assert_size_stride(arg350_1, (24, ), (1, ))
    assert_size_stride(arg351_1, (24, ), (1, ))
    assert_size_stride(arg352_1, (48, ), (1, ))
    assert_size_stride(arg353_1, (48, ), (1, ))
    assert_size_stride(arg354_1, (48, ), (1, ))
    assert_size_stride(arg355_1, (48, ), (1, ))
    assert_size_stride(arg356_1, (24, ), (1, ))
    assert_size_stride(arg357_1, (24, ), (1, ))
    assert_size_stride(arg358_1, (48, ), (1, ))
    assert_size_stride(arg359_1, (48, ), (1, ))
    assert_size_stride(arg360_1, (48, ), (1, ))
    assert_size_stride(arg361_1, (48, ), (1, ))
    assert_size_stride(arg362_1, (24, ), (1, ))
    assert_size_stride(arg363_1, (24, ), (1, ))
    assert_size_stride(arg364_1, (48, ), (1, ))
    assert_size_stride(arg365_1, (48, ), (1, ))
    assert_size_stride(arg366_1, (48, ), (1, ))
    assert_size_stride(arg367_1, (48, ), (1, ))
    assert_size_stride(arg368_1, (24, ), (1, ))
    assert_size_stride(arg369_1, (24, ), (1, ))
    assert_size_stride(arg370_1, (120, ), (1, ))
    assert_size_stride(arg371_1, (120, ), (1, ))
    assert_size_stride(arg372_1, (120, ), (1, ))
    assert_size_stride(arg373_1, (120, ), (1, ))
    assert_size_stride(arg374_1, (40, ), (1, ))
    assert_size_stride(arg375_1, (40, ), (1, ))
    assert_size_stride(arg376_1, (120, ), (1, ))
    assert_size_stride(arg377_1, (120, ), (1, ))
    assert_size_stride(arg378_1, (120, ), (1, ))
    assert_size_stride(arg379_1, (120, ), (1, ))
    assert_size_stride(arg380_1, (40, ), (1, ))
    assert_size_stride(arg381_1, (40, ), (1, ))
    assert_size_stride(arg382_1, (120, ), (1, ))
    assert_size_stride(arg383_1, (120, ), (1, ))
    assert_size_stride(arg384_1, (120, ), (1, ))
    assert_size_stride(arg385_1, (120, ), (1, ))
    assert_size_stride(arg386_1, (40, ), (1, ))
    assert_size_stride(arg387_1, (40, ), (1, ))
    assert_size_stride(arg388_1, (120, ), (1, ))
    assert_size_stride(arg389_1, (120, ), (1, ))
    assert_size_stride(arg390_1, (120, ), (1, ))
    assert_size_stride(arg391_1, (120, ), (1, ))
    assert_size_stride(arg392_1, (40, ), (1, ))
    assert_size_stride(arg393_1, (40, ), (1, ))
    assert_size_stride(arg394_1, (120, ), (1, ))
    assert_size_stride(arg395_1, (120, ), (1, ))
    assert_size_stride(arg396_1, (120, ), (1, ))
    assert_size_stride(arg397_1, (120, ), (1, ))
    assert_size_stride(arg398_1, (40, ), (1, ))
    assert_size_stride(arg399_1, (40, ), (1, ))
    assert_size_stride(arg400_1, (200, ), (1, ))
    assert_size_stride(arg401_1, (200, ), (1, ))
    assert_size_stride(arg402_1, (200, ), (1, ))
    assert_size_stride(arg403_1, (200, ), (1, ))
    assert_size_stride(arg404_1, (72, ), (1, ))
    assert_size_stride(arg405_1, (72, ), (1, ))
    assert_size_stride(arg406_1, (216, ), (1, ))
    assert_size_stride(arg407_1, (216, ), (1, ))
    assert_size_stride(arg408_1, (216, ), (1, ))
    assert_size_stride(arg409_1, (216, ), (1, ))
    assert_size_stride(arg410_1, (72, ), (1, ))
    assert_size_stride(arg411_1, (72, ), (1, ))
    assert_size_stride(arg412_1, (216, ), (1, ))
    assert_size_stride(arg413_1, (216, ), (1, ))
    assert_size_stride(arg414_1, (216, ), (1, ))
    assert_size_stride(arg415_1, (216, ), (1, ))
    assert_size_stride(arg416_1, (72, ), (1, ))
    assert_size_stride(arg417_1, (72, ), (1, ))
    assert_size_stride(arg418_1, (216, ), (1, ))
    assert_size_stride(arg419_1, (216, ), (1, ))
    assert_size_stride(arg420_1, (216, ), (1, ))
    assert_size_stride(arg421_1, (216, ), (1, ))
    assert_size_stride(arg422_1, (72, ), (1, ))
    assert_size_stride(arg423_1, (72, ), (1, ))
    assert_size_stride(arg424_1, (216, ), (1, ))
    assert_size_stride(arg425_1, (216, ), (1, ))
    assert_size_stride(arg426_1, (216, ), (1, ))
    assert_size_stride(arg427_1, (216, ), (1, ))
    assert_size_stride(arg428_1, (72, ), (1, ))
    assert_size_stride(arg429_1, (72, ), (1, ))
    assert_size_stride(arg430_1, (360, ), (1, ))
    assert_size_stride(arg431_1, (360, ), (1, ))
    assert_size_stride(arg432_1, (360, ), (1, ))
    assert_size_stride(arg433_1, (360, ), (1, ))
    assert_size_stride(arg434_1, (120, ), (1, ))
    assert_size_stride(arg435_1, (120, ), (1, ))
    assert_size_stride(arg436_1, (360, ), (1, ))
    assert_size_stride(arg437_1, (360, ), (1, ))
    assert_size_stride(arg438_1, (360, ), (1, ))
    assert_size_stride(arg439_1, (360, ), (1, ))
    assert_size_stride(arg440_1, (120, ), (1, ))
    assert_size_stride(arg441_1, (120, ), (1, ))
    assert_size_stride(arg442_1, (360, ), (1, ))
    assert_size_stride(arg443_1, (360, ), (1, ))
    assert_size_stride(arg444_1, (360, ), (1, ))
    assert_size_stride(arg445_1, (360, ), (1, ))
    assert_size_stride(arg446_1, (120, ), (1, ))
    assert_size_stride(arg447_1, (120, ), (1, ))
    assert_size_stride(arg448_1, (360, ), (1, ))
    assert_size_stride(arg449_1, (360, ), (1, ))
    assert_size_stride(arg450_1, (360, ), (1, ))
    assert_size_stride(arg451_1, (360, ), (1, ))
    assert_size_stride(arg452_1, (120, ), (1, ))
    assert_size_stride(arg453_1, (120, ), (1, ))
    assert_size_stride(arg454_1, (360, ), (1, ))
    assert_size_stride(arg455_1, (360, ), (1, ))
    assert_size_stride(arg456_1, (360, ), (1, ))
    assert_size_stride(arg457_1, (360, ), (1, ))
    assert_size_stride(arg458_1, (120, ), (1, ))
    assert_size_stride(arg459_1, (120, ), (1, ))
    assert_size_stride(arg460_1, (360, ), (1, ))
    assert_size_stride(arg461_1, (360, ), (1, ))
    assert_size_stride(arg462_1, (360, ), (1, ))
    assert_size_stride(arg463_1, (360, ), (1, ))
    assert_size_stride(arg464_1, (120, ), (1, ))
    assert_size_stride(arg465_1, (120, ), (1, ))
    assert_size_stride(arg466_1, (720, ), (1, ))
    assert_size_stride(arg467_1, (720, ), (1, ))
    assert_size_stride(arg468_1, (720, ), (1, ))
    assert_size_stride(arg469_1, (720, ), (1, ))
    assert_size_stride(arg470_1, (184, ), (1, ))
    assert_size_stride(arg471_1, (184, ), (1, ))
    assert_size_stride(arg472_1, (736, ), (1, ))
    assert_size_stride(arg473_1, (736, ), (1, ))
    assert_size_stride(arg474_1, (736, ), (1, ))
    assert_size_stride(arg475_1, (736, ), (1, ))
    assert_size_stride(arg476_1, (184, ), (1, ))
    assert_size_stride(arg477_1, (184, ), (1, ))
    assert_size_stride(arg478_1, (736, ), (1, ))
    assert_size_stride(arg479_1, (736, ), (1, ))
    assert_size_stride(arg480_1, (736, ), (1, ))
    assert_size_stride(arg481_1, (736, ), (1, ))
    assert_size_stride(arg482_1, (184, ), (1, ))
    assert_size_stride(arg483_1, (184, ), (1, ))
    assert_size_stride(arg484_1, (736, ), (1, ))
    assert_size_stride(arg485_1, (736, ), (1, ))
    assert_size_stride(arg486_1, (736, ), (1, ))
    assert_size_stride(arg487_1, (736, ), (1, ))
    assert_size_stride(arg488_1, (184, ), (1, ))
    assert_size_stride(arg489_1, (184, ), (1, ))
    assert_size_stride(arg490_1, (736, ), (1, ))
    assert_size_stride(arg491_1, (736, ), (1, ))
    assert_size_stride(arg492_1, (736, ), (1, ))
    assert_size_stride(arg493_1, (736, ), (1, ))
    assert_size_stride(arg494_1, (184, ), (1, ))
    assert_size_stride(arg495_1, (184, ), (1, ))
    assert_size_stride(arg496_1, (736, ), (1, ))
    assert_size_stride(arg497_1, (736, ), (1, ))
    assert_size_stride(arg498_1, (736, ), (1, ))
    assert_size_stride(arg499_1, (736, ), (1, ))
    assert_size_stride(arg500_1, (184, ), (1, ))
    assert_size_stride(arg501_1, (184, ), (1, ))
    assert_size_stride(arg502_1, (1104, ), (1, ))
    assert_size_stride(arg503_1, (1104, ), (1, ))
    assert_size_stride(arg504_1, (1104, ), (1, ))
    assert_size_stride(arg505_1, (1104, ), (1, ))
    assert_size_stride(arg506_1, (224, ), (1, ))
    assert_size_stride(arg507_1, (224, ), (1, ))
    assert_size_stride(arg508_1, (1344, ), (1, ))
    assert_size_stride(arg509_1, (1344, ), (1, ))
    assert_size_stride(arg510_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg510_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg176_1
    del arg510_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_1(c_void_p(buf4.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg336_1
    del arg337_1
    # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.hardswish]
    buf5 = extern_kernels.convolution(buf4, arg177_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf5, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del arg177_1
    buf6 = buf5; del buf5  # reuse
    buf7 = buf6; del buf6  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_2(c_void_p(buf7.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg2_1
    del arg338_1
    del arg339_1
    del arg3_1
    # Source Nodes: [x_11, x_9], Original ATen: [aten.convolution, aten.hardswish]
    buf8 = extern_kernels.convolution(buf7, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del arg178_1
    del buf7
    buf9 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_3(c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg340_1
    del arg341_1
    del arg4_1
    del arg5_1
    del buf8
    # Source Nodes: [x_17], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf9, arg179_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf10, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del arg179_1
    buf11 = buf10; del buf10  # reuse
    buf12 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_4(c_void_p(buf12.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg342_1
    del arg343_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_21, x_23], Original ATen: [aten.convolution, aten.hardswish]
    buf13 = extern_kernels.convolution(buf12, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del arg180_1
    del buf12
    buf14 = buf13; del buf13  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_5(c_void_p(buf14.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg344_1
    del arg345_1
    del arg8_1
    del arg9_1
    del buf9
    # Source Nodes: [shortcut_2, x_24, x_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf15 = extern_kernels.convolution(buf14, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del arg181_1
    del buf14
    buf16 = buf15; del buf15  # reuse
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_6(c_void_p(buf17.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg346_1
    del arg347_1
    # Source Nodes: [x_33, x_34], Original ATen: [aten.convolution, aten.hardswish]
    buf18 = extern_kernels.convolution(buf17, arg182_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf18, (8, 64, 64, 64), (262144, 1, 4096, 64))
    del arg182_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    buf20 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_7(c_void_p(buf20.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    del arg348_1
    del arg349_1
    # Source Nodes: [x_38, x_40], Original ATen: [aten.convolution, aten.hardswish]
    buf21 = extern_kernels.convolution(buf20, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 24, 64, 64), (98304, 1, 1536, 24))
    del arg183_1
    del buf20
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_8(c_void_p(buf22.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg350_1
    del arg351_1
    # Source Nodes: [x_45], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(buf22, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg184_1
    buf24 = buf23; del buf23  # reuse
    buf25 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_9(c_void_p(buf25.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg352_1
    del arg353_1
    # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.hardswish]
    buf26 = extern_kernels.convolution(buf25, arg185_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf26, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg185_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    buf28 = buf27; del buf27  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_10(c_void_p(buf28.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg354_1
    del arg355_1
    # Source Nodes: [x_54, x_56], Original ATen: [aten.convolution, aten.hardswish]
    buf29 = extern_kernels.convolution(buf28, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 24, 64, 64), (98304, 1, 1536, 24))
    del arg186_1
    del buf28
    buf30 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_11(c_void_p(buf30.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg356_1
    del arg357_1
    del buf29
    # Source Nodes: [x_62], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg187_1
    buf32 = buf31; del buf31  # reuse
    buf33 = buf32; del buf32  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_12(c_void_p(buf33.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg358_1
    del arg359_1
    # Source Nodes: [x_66, x_67], Original ATen: [aten.convolution, aten.hardswish]
    buf34 = extern_kernels.convolution(buf33, arg188_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf34, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg188_1
    del buf33
    buf35 = buf34; del buf34  # reuse
    buf36 = buf35; del buf35  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_13(c_void_p(buf36.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg24_1
    del arg25_1
    del arg360_1
    del arg361_1
    # Source Nodes: [x_71, x_73], Original ATen: [aten.convolution, aten.hardswish]
    buf37 = extern_kernels.convolution(buf36, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 24, 64, 64), (98304, 1, 1536, 24))
    del arg189_1
    del buf36
    buf38 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_14(c_void_p(buf38.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    del arg362_1
    del arg363_1
    del buf37
    # Source Nodes: [x_79], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg190_1
    buf40 = buf39; del buf39  # reuse
    buf41 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_15(c_void_p(buf41.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg28_1
    del arg29_1
    del arg364_1
    del arg365_1
    # Source Nodes: [x_83, x_84], Original ATen: [aten.convolution, aten.hardswish]
    buf42 = extern_kernels.convolution(buf41, arg191_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf42, (8, 48, 64, 64), (196608, 1, 3072, 48))
    del arg191_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    buf44 = buf43; del buf43  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_16(c_void_p(buf44.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg30_1
    del arg31_1
    del arg366_1
    del arg367_1
    # Source Nodes: [x_88, x_90], Original ATen: [aten.convolution, aten.hardswish]
    buf45 = extern_kernels.convolution(buf44, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (8, 24, 64, 64), (98304, 1, 1536, 24))
    del arg192_1
    del buf44
    buf46 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_17(c_void_p(buf46.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg32_1
    del arg33_1
    del arg368_1
    del arg369_1
    del buf45
    # Source Nodes: [shortcut_6, x_91, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf47 = extern_kernels.convolution(buf46, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (8, 120, 64, 64), (491520, 1, 7680, 120))
    del arg193_1
    del buf46
    buf48 = buf47; del buf47  # reuse
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_18(c_void_p(buf49.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg34_1
    del arg35_1
    del arg370_1
    del arg371_1
    # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.hardswish]
    buf50 = extern_kernels.convolution(buf49, arg194_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf50, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg194_1
    del buf49
    buf51 = buf50; del buf50  # reuse
    buf52 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf52, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_19(c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg36_1
    del arg372_1
    del arg373_1
    del arg37_1
    # Source Nodes: [x_105, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf54 = extern_kernels.convolution(buf53, arg195_1, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf54, (8, 8, 1, 1), (8, 1, 8, 8))
    del arg195_1
    del arg196_1
    del buf53
    buf55 = buf54; del buf54  # reuse
    cpp_fused_hardswish_20(c_void_p(buf55.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardswish]
    buf56 = extern_kernels.convolution(buf55, arg197_1, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf56, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg197_1
    del arg198_1
    del buf55
    buf57 = buf51; del buf51  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_21(c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_105, x_106, x_107], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf58 = extern_kernels.convolution(buf57, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (8, 40, 32, 32), (40960, 1, 1280, 40))
    del arg199_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_22(c_void_p(buf59.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg374_1
    del arg375_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_112], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg200_1
    buf61 = buf60; del buf60  # reuse
    buf62 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_23(c_void_p(buf62.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg376_1
    del arg377_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_116, x_117], Original ATen: [aten.convolution, aten.hardswish]
    buf63 = extern_kernels.convolution(buf62, arg201_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf63, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg201_1
    del buf62
    buf64 = buf63; del buf63  # reuse
    buf65 = reinterpret_tensor(buf56, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf56  # reuse
    buf66 = reinterpret_tensor(buf65, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_24(c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg378_1
    del arg379_1
    del arg42_1
    del arg43_1
    # Source Nodes: [x_121, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf67 = extern_kernels.convolution(buf66, arg202_1, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf67, (8, 16, 1, 1), (16, 1, 16, 16))
    del arg202_1
    del arg203_1
    del buf66
    buf68 = buf67; del buf67  # reuse
    cpp_fused_hardswish_25(c_void_p(buf68.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardswish]
    buf69 = extern_kernels.convolution(buf68, arg204_1, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf69, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg204_1
    del arg205_1
    del buf68
    buf70 = buf64; del buf64  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_26(c_void_p(buf70.data_ptr()), c_void_p(buf69.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_121, x_122, x_123], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf71 = extern_kernels.convolution(buf70, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (8, 40, 32, 32), (40960, 1, 1280, 40))
    del arg206_1
    del buf70
    buf72 = buf59; del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_27(c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg380_1
    del arg381_1
    del arg44_1
    del arg45_1
    del buf71
    # Source Nodes: [x_129], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg207_1
    buf74 = buf73; del buf73  # reuse
    buf75 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_28(c_void_p(buf75.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg382_1
    del arg383_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_133, x_134], Original ATen: [aten.convolution, aten.hardswish]
    buf76 = extern_kernels.convolution(buf75, arg208_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf76, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg208_1
    del buf75
    buf77 = buf76; del buf76  # reuse
    buf78 = reinterpret_tensor(buf69, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf69  # reuse
    buf79 = reinterpret_tensor(buf78, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_29(c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg384_1
    del arg385_1
    del arg48_1
    del arg49_1
    # Source Nodes: [x_138, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf80 = extern_kernels.convolution(buf79, arg209_1, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf80, (8, 16, 1, 1), (16, 1, 16, 16))
    del arg209_1
    del arg210_1
    del buf79
    buf81 = buf80; del buf80  # reuse
    cpp_fused_hardswish_30(c_void_p(buf81.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.hardswish]
    buf82 = extern_kernels.convolution(buf81, arg211_1, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg211_1
    del arg212_1
    del buf81
    buf83 = buf77; del buf77  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_31(c_void_p(buf83.data_ptr()), c_void_p(buf82.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_138, x_139, x_140], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf84 = extern_kernels.convolution(buf83, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 40, 32, 32), (40960, 1, 1280, 40))
    del arg213_1
    del buf83
    buf85 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_32(c_void_p(buf85.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg386_1
    del arg387_1
    del arg50_1
    del arg51_1
    del buf84
    # Source Nodes: [x_146], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg214_1
    buf87 = buf86; del buf86  # reuse
    buf88 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_33(c_void_p(buf88.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg388_1
    del arg389_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_150, x_151], Original ATen: [aten.convolution, aten.hardswish]
    buf89 = extern_kernels.convolution(buf88, arg215_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf89, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg215_1
    del buf88
    buf90 = buf89; del buf89  # reuse
    buf91 = reinterpret_tensor(buf82, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf82  # reuse
    buf92 = reinterpret_tensor(buf91, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_34(c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg390_1
    del arg391_1
    del arg54_1
    del arg55_1
    # Source Nodes: [x_155, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf93 = extern_kernels.convolution(buf92, arg216_1, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf93, (8, 16, 1, 1), (16, 1, 16, 16))
    del arg216_1
    del arg217_1
    del buf92
    buf94 = buf93; del buf93  # reuse
    cpp_fused_hardswish_35(c_void_p(buf94.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardswish]
    buf95 = extern_kernels.convolution(buf94, arg218_1, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf95, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg218_1
    del arg219_1
    del buf94
    buf96 = buf90; del buf90  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_36(c_void_p(buf96.data_ptr()), c_void_p(buf95.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_155, x_156, x_157], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf97 = extern_kernels.convolution(buf96, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf97, (8, 40, 32, 32), (40960, 1, 1280, 40))
    del arg220_1
    del buf96
    buf98 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_37(c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg392_1
    del arg393_1
    del arg56_1
    del arg57_1
    del buf97
    # Source Nodes: [x_163], Original ATen: [aten.convolution]
    buf99 = extern_kernels.convolution(buf98, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg221_1
    buf100 = buf99; del buf99  # reuse
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_38(c_void_p(buf101.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg394_1
    del arg395_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_167, x_168], Original ATen: [aten.convolution, aten.hardswish]
    buf102 = extern_kernels.convolution(buf101, arg222_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf102, (8, 120, 32, 32), (122880, 1, 3840, 120))
    del arg222_1
    del buf101
    buf103 = buf102; del buf102  # reuse
    buf104 = reinterpret_tensor(buf95, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf95  # reuse
    buf105 = reinterpret_tensor(buf104, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_39(c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg396_1
    del arg397_1
    del arg60_1
    del arg61_1
    # Source Nodes: [x_172, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf106 = extern_kernels.convolution(buf105, arg223_1, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (8, 16, 1, 1), (16, 1, 16, 16))
    del arg223_1
    del arg224_1
    del buf105
    buf107 = buf106; del buf106  # reuse
    cpp_fused_hardswish_40(c_void_p(buf107.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardswish]
    buf108 = extern_kernels.convolution(buf107, arg225_1, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf108, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg225_1
    del arg226_1
    del buf107
    buf109 = buf103; del buf103  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_41(c_void_p(buf109.data_ptr()), c_void_p(buf108.data_ptr()))
    del buf108
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____4___se_gate, x_172, x_173, x_174], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf110 = extern_kernels.convolution(buf109, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (8, 40, 32, 32), (40960, 1, 1280, 40))
    del arg227_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_42(c_void_p(buf111.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg398_1
    del arg399_1
    del arg62_1
    del arg63_1
    del buf98
    # Source Nodes: [shortcut_11, x_175, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf112 = extern_kernels.convolution(buf111, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf112, (8, 200, 32, 32), (204800, 1, 6400, 200))
    del arg228_1
    del buf111
    buf113 = buf112; del buf112  # reuse
    buf114 = buf113; del buf113  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_43(c_void_p(buf114.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg400_1
    del arg401_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_184, x_185], Original ATen: [aten.convolution, aten.hardswish]
    buf115 = extern_kernels.convolution(buf114, arg229_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
    assert_size_stride(buf115, (8, 200, 16, 16), (51200, 1, 3200, 200))
    del arg229_1
    del buf114
    buf116 = buf115; del buf115  # reuse
    buf117 = buf116; del buf116  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_44(c_void_p(buf117.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg402_1
    del arg403_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_189, x_191], Original ATen: [aten.convolution, aten.hardswish]
    buf118 = extern_kernels.convolution(buf117, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (8, 72, 16, 16), (18432, 1, 1152, 72))
    del arg230_1
    del buf117
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_45(c_void_p(buf119.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg404_1
    del arg405_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_196], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg231_1
    buf121 = buf120; del buf120  # reuse
    buf122 = buf121; del buf121  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_46(c_void_p(buf122.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg406_1
    del arg407_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_200, x_201], Original ATen: [aten.convolution, aten.hardswish]
    buf123 = extern_kernels.convolution(buf122, arg232_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
    assert_size_stride(buf123, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg232_1
    del buf122
    buf124 = buf123; del buf123  # reuse
    buf125 = buf124; del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_47(c_void_p(buf125.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg408_1
    del arg409_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_205, x_207], Original ATen: [aten.convolution, aten.hardswish]
    buf126 = extern_kernels.convolution(buf125, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 72, 16, 16), (18432, 1, 1152, 72))
    del arg233_1
    del buf125
    buf127 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_48(c_void_p(buf127.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg410_1
    del arg411_1
    del arg74_1
    del arg75_1
    del buf126
    # Source Nodes: [x_213], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg234_1
    buf129 = buf128; del buf128  # reuse
    buf130 = buf129; del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_49(c_void_p(buf130.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg412_1
    del arg413_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_217, x_218], Original ATen: [aten.convolution, aten.hardswish]
    buf131 = extern_kernels.convolution(buf130, arg235_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
    assert_size_stride(buf131, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg235_1
    del buf130
    buf132 = buf131; del buf131  # reuse
    buf133 = buf132; del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_50(c_void_p(buf133.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg414_1
    del arg415_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_222, x_224], Original ATen: [aten.convolution, aten.hardswish]
    buf134 = extern_kernels.convolution(buf133, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (8, 72, 16, 16), (18432, 1, 1152, 72))
    del arg236_1
    del buf133
    buf135 = buf127; del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_51(c_void_p(buf135.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg416_1
    del arg417_1
    del arg80_1
    del arg81_1
    del buf134
    # Source Nodes: [x_230], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf136, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg237_1
    buf137 = buf136; del buf136  # reuse
    buf138 = buf137; del buf137  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_52(c_void_p(buf138.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg418_1
    del arg419_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_234, x_235], Original ATen: [aten.convolution, aten.hardswish]
    buf139 = extern_kernels.convolution(buf138, arg238_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
    assert_size_stride(buf139, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg238_1
    del buf138
    buf140 = buf139; del buf139  # reuse
    buf141 = buf140; del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_53(c_void_p(buf141.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg420_1
    del arg421_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_239, x_241], Original ATen: [aten.convolution, aten.hardswish]
    buf142 = extern_kernels.convolution(buf141, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (8, 72, 16, 16), (18432, 1, 1152, 72))
    del arg239_1
    del buf141
    buf143 = buf135; del buf135  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_54(c_void_p(buf143.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg423_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg422_1
    del arg423_1
    del arg86_1
    del arg87_1
    del buf142
    # Source Nodes: [x_247], Original ATen: [aten.convolution]
    buf144 = extern_kernels.convolution(buf143, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf144, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg240_1
    buf145 = buf144; del buf144  # reuse
    buf146 = buf145; del buf145  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_55(c_void_p(buf146.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg424_1
    del arg425_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_251, x_252], Original ATen: [aten.convolution, aten.hardswish]
    buf147 = extern_kernels.convolution(buf146, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
    assert_size_stride(buf147, (8, 216, 16, 16), (55296, 1, 3456, 216))
    del arg241_1
    del buf146
    buf148 = buf147; del buf147  # reuse
    buf149 = buf148; del buf148  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_56(c_void_p(buf149.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg426_1
    del arg427_1
    del arg90_1
    del arg91_1
    # Source Nodes: [x_256, x_258], Original ATen: [aten.convolution, aten.hardswish]
    buf150 = extern_kernels.convolution(buf149, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf150, (8, 72, 16, 16), (18432, 1, 1152, 72))
    del arg242_1
    del buf149
    buf151 = buf143; del buf143  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_57(c_void_p(buf151.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg429_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg428_1
    del arg429_1
    del arg92_1
    del arg93_1
    del buf150
    # Source Nodes: [shortcut_16, x_259, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf152 = extern_kernels.convolution(buf151, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg243_1
    del buf151
    buf153 = buf152; del buf152  # reuse
    buf154 = buf153; del buf153  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_58(c_void_p(buf154.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg430_1
    del arg431_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_268, x_269], Original ATen: [aten.convolution, aten.hardswish]
    buf155 = extern_kernels.convolution(buf154, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf155, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg244_1
    del buf154
    buf156 = buf155; del buf155  # reuse
    buf157 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cpu', dtype=torch.float32)
    buf158 = reinterpret_tensor(buf157, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf157  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_59(c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg432_1
    del arg433_1
    del arg96_1
    del arg97_1
    # Source Nodes: [x_273, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf159 = extern_kernels.convolution(buf158, arg245_1, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf159, (8, 24, 1, 1), (24, 1, 24, 24))
    del arg245_1
    del arg246_1
    del buf158
    buf160 = buf159; del buf159  # reuse
    cpp_fused_hardswish_60(c_void_p(buf160.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardswish]
    buf161 = extern_kernels.convolution(buf160, arg247_1, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf161, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg247_1
    del arg248_1
    del buf160
    buf162 = buf156; del buf156  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_61(c_void_p(buf162.data_ptr()), c_void_p(buf161.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_273, x_274, x_275], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf163 = extern_kernels.convolution(buf162, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf163, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg249_1
    del buf162
    buf164 = buf163; del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_62(c_void_p(buf164.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg435_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg434_1
    del arg435_1
    del arg98_1
    del arg99_1
    # Source Nodes: [x_280], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(buf164, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf165, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg250_1
    buf166 = buf165; del buf165  # reuse
    buf167 = buf166; del buf166  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_63(c_void_p(buf167.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg436_1
    del arg437_1
    # Source Nodes: [x_284, x_285], Original ATen: [aten.convolution, aten.hardswish]
    buf168 = extern_kernels.convolution(buf167, arg251_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf168, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg251_1
    del buf167
    buf169 = buf168; del buf168  # reuse
    buf170 = reinterpret_tensor(buf161, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf161  # reuse
    buf171 = reinterpret_tensor(buf170, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf170  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_64(c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg438_1
    del arg439_1
    # Source Nodes: [x_289, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf172 = extern_kernels.convolution(buf171, arg252_1, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf172, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg252_1
    del arg253_1
    del buf171
    buf173 = buf172; del buf172  # reuse
    cpp_fused_hardswish_65(c_void_p(buf173.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardswish]
    buf174 = extern_kernels.convolution(buf173, arg254_1, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf174, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg254_1
    del arg255_1
    del buf173
    buf175 = buf169; del buf169  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_66(c_void_p(buf175.data_ptr()), c_void_p(buf174.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_289, x_290, x_291], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf176 = extern_kernels.convolution(buf175, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf176, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg256_1
    del buf175
    buf177 = buf164; del buf164  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_67(c_void_p(buf177.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()))
    del arg104_1
    del arg105_1
    del arg440_1
    del arg441_1
    del buf176
    # Source Nodes: [x_297], Original ATen: [aten.convolution]
    buf178 = extern_kernels.convolution(buf177, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf178, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg257_1
    buf179 = buf178; del buf178  # reuse
    buf180 = buf179; del buf179  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_68(c_void_p(buf180.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg442_1
    del arg443_1
    # Source Nodes: [x_301, x_302], Original ATen: [aten.convolution, aten.hardswish]
    buf181 = extern_kernels.convolution(buf180, arg258_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf181, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg258_1
    del buf180
    buf182 = buf181; del buf181  # reuse
    buf183 = reinterpret_tensor(buf174, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf174  # reuse
    buf184 = reinterpret_tensor(buf183, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf183  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_69(c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg444_1
    del arg445_1
    # Source Nodes: [x_306, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf185 = extern_kernels.convolution(buf184, arg259_1, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf185, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg259_1
    del arg260_1
    del buf184
    buf186 = buf185; del buf185  # reuse
    cpp_fused_hardswish_70(c_void_p(buf186.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardswish]
    buf187 = extern_kernels.convolution(buf186, arg261_1, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf187, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg261_1
    del arg262_1
    del buf186
    buf188 = buf182; del buf182  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_71(c_void_p(buf188.data_ptr()), c_void_p(buf187.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_306, x_307, x_308], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf189 = extern_kernels.convolution(buf188, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf189, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg263_1
    del buf188
    buf190 = buf177; del buf177  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_72(c_void_p(buf190.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg446_1
    del arg447_1
    del buf189
    # Source Nodes: [x_314], Original ATen: [aten.convolution]
    buf191 = extern_kernels.convolution(buf190, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf191, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg264_1
    buf192 = buf191; del buf191  # reuse
    buf193 = buf192; del buf192  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_73(c_void_p(buf193.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg448_1
    del arg449_1
    # Source Nodes: [x_318, x_319], Original ATen: [aten.convolution, aten.hardswish]
    buf194 = extern_kernels.convolution(buf193, arg265_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf194, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg265_1
    del buf193
    buf195 = buf194; del buf194  # reuse
    buf196 = reinterpret_tensor(buf187, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf187  # reuse
    buf197 = reinterpret_tensor(buf196, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf196  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_74(c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg450_1
    del arg451_1
    # Source Nodes: [x_323, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf198 = extern_kernels.convolution(buf197, arg266_1, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf198, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg266_1
    del arg267_1
    del buf197
    buf199 = buf198; del buf198  # reuse
    cpp_fused_hardswish_75(c_void_p(buf199.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.hardswish]
    buf200 = extern_kernels.convolution(buf199, arg268_1, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf200, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg268_1
    del arg269_1
    del buf199
    buf201 = buf195; del buf195  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_76(c_void_p(buf201.data_ptr()), c_void_p(buf200.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_323, x_324, x_325], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf202 = extern_kernels.convolution(buf201, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf202, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg270_1
    del buf201
    buf203 = buf190; del buf190  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_77(c_void_p(buf203.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()))
    del arg116_1
    del arg117_1
    del arg452_1
    del arg453_1
    del buf202
    # Source Nodes: [x_331], Original ATen: [aten.convolution]
    buf204 = extern_kernels.convolution(buf203, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf204, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg271_1
    buf205 = buf204; del buf204  # reuse
    buf206 = buf205; del buf205  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_78(c_void_p(buf206.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg454_1
    del arg455_1
    # Source Nodes: [x_335, x_336], Original ATen: [aten.convolution, aten.hardswish]
    buf207 = extern_kernels.convolution(buf206, arg272_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf207, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg272_1
    del buf206
    buf208 = buf207; del buf207  # reuse
    buf209 = reinterpret_tensor(buf200, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf200  # reuse
    buf210 = reinterpret_tensor(buf209, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_79(c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg456_1
    del arg457_1
    # Source Nodes: [x_340, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf211 = extern_kernels.convolution(buf210, arg273_1, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf211, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg273_1
    del arg274_1
    del buf210
    buf212 = buf211; del buf211  # reuse
    cpp_fused_hardswish_80(c_void_p(buf212.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.hardswish]
    buf213 = extern_kernels.convolution(buf212, arg275_1, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf213, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg275_1
    del arg276_1
    del buf212
    buf214 = buf208; del buf208  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_81(c_void_p(buf214.data_ptr()), c_void_p(buf213.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____4___se_gate, x_340, x_341, x_342], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf215 = extern_kernels.convolution(buf214, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf215, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg277_1
    del buf214
    buf216 = buf203; del buf203  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_82(c_void_p(buf216.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg459_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg122_1
    del arg123_1
    del arg458_1
    del arg459_1
    del buf215
    # Source Nodes: [x_348], Original ATen: [aten.convolution]
    buf217 = extern_kernels.convolution(buf216, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf217, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg278_1
    buf218 = buf217; del buf217  # reuse
    buf219 = buf218; del buf218  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_83(c_void_p(buf219.data_ptr()), c_void_p(arg460_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg460_1
    del arg461_1
    # Source Nodes: [x_352, x_353], Original ATen: [aten.convolution, aten.hardswish]
    buf220 = extern_kernels.convolution(buf219, arg279_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
    assert_size_stride(buf220, (8, 360, 16, 16), (92160, 1, 5760, 360))
    del arg279_1
    del buf219
    buf221 = buf220; del buf220  # reuse
    buf222 = reinterpret_tensor(buf213, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf213  # reuse
    buf223 = reinterpret_tensor(buf222, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf222  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_84(c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    del arg462_1
    del arg463_1
    # Source Nodes: [x_357, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf224 = extern_kernels.convolution(buf223, arg280_1, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf224, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg280_1
    del arg281_1
    del buf223
    buf225 = buf224; del buf224  # reuse
    cpp_fused_hardswish_85(c_void_p(buf225.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.hardswish]
    buf226 = extern_kernels.convolution(buf225, arg282_1, arg283_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf226, (8, 360, 1, 1), (360, 1, 360, 360))
    del arg282_1
    del arg283_1
    del buf225
    buf227 = buf221; del buf221  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_86(c_void_p(buf227.data_ptr()), c_void_p(buf226.data_ptr()))
    del buf226
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____5___se_gate, x_357, x_358, x_359], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf228 = extern_kernels.convolution(buf227, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (8, 120, 16, 16), (30720, 1, 1920, 120))
    del arg284_1
    del buf227
    buf229 = buf216; del buf216  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_87(c_void_p(buf229.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg465_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()))
    del arg128_1
    del arg129_1
    del arg464_1
    del arg465_1
    del buf228
    # Source Nodes: [shortcut_22, x_360, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf230 = extern_kernels.convolution(buf229, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf230, (8, 720, 16, 16), (184320, 1, 11520, 720))
    del arg285_1
    del buf229
    buf231 = buf230; del buf230  # reuse
    buf232 = buf231; del buf231  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_88(c_void_p(buf232.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()))
    del arg130_1
    del arg131_1
    del arg466_1
    del arg467_1
    # Source Nodes: [x_369, x_370], Original ATen: [aten.convolution, aten.hardswish]
    buf233 = extern_kernels.convolution(buf232, arg286_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=720, bias=None)
    assert_size_stride(buf233, (8, 720, 8, 8), (46080, 1, 5760, 720))
    del arg286_1
    del buf232
    buf234 = buf233; del buf233  # reuse
    buf235 = empty_strided((8, 720, 1, 1), (720, 1, 5760, 5760), device='cpu', dtype=torch.float32)
    buf236 = reinterpret_tensor(buf235, (8, 720, 1, 1), (720, 1, 720, 720), 0); del buf235  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_89(c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg132_1
    del arg133_1
    del arg468_1
    del arg469_1
    # Source Nodes: [x_374, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf237 = extern_kernels.convolution(buf236, arg287_1, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf237, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg287_1
    del arg288_1
    del buf236
    buf238 = buf237; del buf237  # reuse
    cpp_fused_hardswish_90(c_void_p(buf238.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.hardswish]
    buf239 = extern_kernels.convolution(buf238, arg289_1, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf239, (8, 720, 1, 1), (720, 1, 720, 720))
    del arg289_1
    del arg290_1
    del buf238
    buf240 = buf234; del buf234  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_91(c_void_p(buf240.data_ptr()), c_void_p(buf239.data_ptr()))
    del buf239
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_374, x_375, x_376], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf241 = extern_kernels.convolution(buf240, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf241, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg291_1
    del buf240
    buf242 = buf241; del buf241  # reuse
    cpp_fused__native_batch_norm_legit_no_training_92(c_void_p(buf242.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()))
    del arg134_1
    del arg135_1
    del arg470_1
    del arg471_1
    # Source Nodes: [x_381], Original ATen: [aten.convolution]
    buf243 = extern_kernels.convolution(buf242, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf243, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg292_1
    buf244 = buf243; del buf243  # reuse
    buf245 = buf244; del buf244  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_93(c_void_p(buf245.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg136_1
    del arg137_1
    del arg472_1
    del arg473_1
    # Source Nodes: [x_385, x_386], Original ATen: [aten.convolution, aten.hardswish]
    buf246 = extern_kernels.convolution(buf245, arg293_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
    assert_size_stride(buf246, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg293_1
    del buf245
    buf247 = buf246; del buf246  # reuse
    buf248 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cpu', dtype=torch.float32)
    buf249 = reinterpret_tensor(buf248, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf248  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_94(c_void_p(buf247.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(arg474_1.data_ptr()), c_void_p(arg475_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg138_1
    del arg139_1
    del arg474_1
    del arg475_1
    # Source Nodes: [x_390, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf250 = extern_kernels.convolution(buf249, arg294_1, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf250, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg294_1
    del arg295_1
    del buf249
    buf251 = buf250; del buf250  # reuse
    cpp_fused_hardswish_95(c_void_p(buf251.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.hardswish]
    buf252 = extern_kernels.convolution(buf251, arg296_1, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf252, (8, 736, 1, 1), (736, 1, 736, 736))
    del arg296_1
    del arg297_1
    del buf251
    buf253 = buf247; del buf247  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_96(c_void_p(buf253.data_ptr()), c_void_p(buf252.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_390, x_391, x_392], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf254 = extern_kernels.convolution(buf253, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf254, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg298_1
    del buf253
    buf255 = buf242; del buf242  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_97(c_void_p(buf255.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg477_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()))
    del arg140_1
    del arg141_1
    del arg476_1
    del arg477_1
    del buf254
    # Source Nodes: [x_398], Original ATen: [aten.convolution]
    buf256 = extern_kernels.convolution(buf255, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf256, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg299_1
    buf257 = buf256; del buf256  # reuse
    buf258 = buf257; del buf257  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_98(c_void_p(buf258.data_ptr()), c_void_p(arg478_1.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    del arg478_1
    del arg479_1
    # Source Nodes: [x_402, x_403], Original ATen: [aten.convolution, aten.hardswish]
    buf259 = extern_kernels.convolution(buf258, arg300_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
    assert_size_stride(buf259, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg300_1
    del buf258
    buf260 = buf259; del buf259  # reuse
    buf261 = reinterpret_tensor(buf252, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf252  # reuse
    buf262 = reinterpret_tensor(buf261, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf261  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_99(c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg144_1
    del arg145_1
    del arg480_1
    del arg481_1
    # Source Nodes: [x_407, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf263 = extern_kernels.convolution(buf262, arg301_1, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf263, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg301_1
    del arg302_1
    del buf262
    buf264 = buf263; del buf263  # reuse
    cpp_fused_hardswish_100(c_void_p(buf264.data_ptr()))
    # Source Nodes: [x_se_54, x_se_55], Original ATen: [aten.convolution, aten.hardswish]
    buf265 = extern_kernels.convolution(buf264, arg303_1, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf265, (8, 736, 1, 1), (736, 1, 736, 736))
    del arg303_1
    del arg304_1
    del buf264
    buf266 = buf260; del buf260  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_101(c_void_p(buf266.data_ptr()), c_void_p(buf265.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_407, x_408, x_409], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf267 = extern_kernels.convolution(buf266, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf267, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg305_1
    del buf266
    buf268 = buf255; del buf255  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_102(c_void_p(buf268.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg483_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg482_1
    del arg483_1
    del buf267
    # Source Nodes: [x_415], Original ATen: [aten.convolution]
    buf269 = extern_kernels.convolution(buf268, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf269, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg306_1
    buf270 = buf269; del buf269  # reuse
    buf271 = buf270; del buf270  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_103(c_void_p(buf271.data_ptr()), c_void_p(arg484_1.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()))
    del arg148_1
    del arg149_1
    del arg484_1
    del arg485_1
    # Source Nodes: [x_419, x_420], Original ATen: [aten.convolution, aten.hardswish]
    buf272 = extern_kernels.convolution(buf271, arg307_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
    assert_size_stride(buf272, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg307_1
    del buf271
    buf273 = buf272; del buf272  # reuse
    buf274 = reinterpret_tensor(buf265, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf265  # reuse
    buf275 = reinterpret_tensor(buf274, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf274  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_104(c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(arg486_1.data_ptr()), c_void_p(arg487_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()))
    del arg150_1
    del arg151_1
    del arg486_1
    del arg487_1
    # Source Nodes: [x_424, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf276 = extern_kernels.convolution(buf275, arg308_1, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf276, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg308_1
    del arg309_1
    del buf275
    buf277 = buf276; del buf276  # reuse
    cpp_fused_hardswish_105(c_void_p(buf277.data_ptr()))
    # Source Nodes: [x_se_58, x_se_59], Original ATen: [aten.convolution, aten.hardswish]
    buf278 = extern_kernels.convolution(buf277, arg310_1, arg311_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf278, (8, 736, 1, 1), (736, 1, 736, 736))
    del arg310_1
    del arg311_1
    del buf277
    buf279 = buf273; del buf273  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_106(c_void_p(buf279.data_ptr()), c_void_p(buf278.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_424, x_425, x_426], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf280 = extern_kernels.convolution(buf279, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf280, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg312_1
    del buf279
    buf281 = buf268; del buf268  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_107(c_void_p(buf281.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()))
    del arg152_1
    del arg153_1
    del arg488_1
    del arg489_1
    del buf280
    # Source Nodes: [x_432], Original ATen: [aten.convolution]
    buf282 = extern_kernels.convolution(buf281, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf282, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg313_1
    buf283 = buf282; del buf282  # reuse
    buf284 = buf283; del buf283  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_108(c_void_p(buf284.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()))
    del arg154_1
    del arg155_1
    del arg490_1
    del arg491_1
    # Source Nodes: [x_436, x_437], Original ATen: [aten.convolution, aten.hardswish]
    buf285 = extern_kernels.convolution(buf284, arg314_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
    assert_size_stride(buf285, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg314_1
    del buf284
    buf286 = buf285; del buf285  # reuse
    buf287 = reinterpret_tensor(buf278, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf278  # reuse
    buf288 = reinterpret_tensor(buf287, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf287  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_109(c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(arg492_1.data_ptr()), c_void_p(arg493_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()))
    del arg156_1
    del arg157_1
    del arg492_1
    del arg493_1
    # Source Nodes: [x_441, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf289 = extern_kernels.convolution(buf288, arg315_1, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf289, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg315_1
    del arg316_1
    del buf288
    buf290 = buf289; del buf289  # reuse
    cpp_fused_hardswish_110(c_void_p(buf290.data_ptr()))
    # Source Nodes: [x_se_62, x_se_63], Original ATen: [aten.convolution, aten.hardswish]
    buf291 = extern_kernels.convolution(buf290, arg317_1, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf291, (8, 736, 1, 1), (736, 1, 736, 736))
    del arg317_1
    del arg318_1
    del buf290
    buf292 = buf286; del buf286  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_111(c_void_p(buf292.data_ptr()), c_void_p(buf291.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_441, x_442, x_443], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf293 = extern_kernels.convolution(buf292, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf293, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg319_1
    del buf292
    buf294 = buf281; del buf281  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_112(c_void_p(buf294.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg495_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()))
    del arg158_1
    del arg159_1
    del arg494_1
    del arg495_1
    del buf293
    # Source Nodes: [x_449], Original ATen: [aten.convolution]
    buf295 = extern_kernels.convolution(buf294, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf295, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg320_1
    buf296 = buf295; del buf295  # reuse
    buf297 = buf296; del buf296  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_113(c_void_p(buf297.data_ptr()), c_void_p(arg496_1.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()))
    del arg160_1
    del arg161_1
    del arg496_1
    del arg497_1
    # Source Nodes: [x_453, x_454], Original ATen: [aten.convolution, aten.hardswish]
    buf298 = extern_kernels.convolution(buf297, arg321_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
    assert_size_stride(buf298, (8, 736, 8, 8), (47104, 1, 5888, 736))
    del arg321_1
    del buf297
    buf299 = buf298; del buf298  # reuse
    buf300 = reinterpret_tensor(buf291, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf291  # reuse
    buf301 = reinterpret_tensor(buf300, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf300  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_114(c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(arg498_1.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg498_1
    del arg499_1
    # Source Nodes: [x_458, x_se_64, x_se_65], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf302 = extern_kernels.convolution(buf301, arg322_1, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf302, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg322_1
    del arg323_1
    del buf301
    buf303 = buf302; del buf302  # reuse
    cpp_fused_hardswish_115(c_void_p(buf303.data_ptr()))
    # Source Nodes: [x_se_66, x_se_67], Original ATen: [aten.convolution, aten.hardswish]
    buf304 = extern_kernels.convolution(buf303, arg324_1, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf304, (8, 736, 1, 1), (736, 1, 736, 736))
    del arg324_1
    del arg325_1
    del buf303
    buf305 = buf299; del buf299  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_116(c_void_p(buf305.data_ptr()), c_void_p(buf304.data_ptr()))
    del buf304
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____5___se_gate, x_458, x_459, x_460], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf306 = extern_kernels.convolution(buf305, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf306, (8, 184, 8, 8), (11776, 1, 1472, 184))
    del arg326_1
    del buf305
    buf307 = buf294; del buf294  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_117(c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg501_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()))
    del arg164_1
    del arg165_1
    del arg500_1
    del arg501_1
    del buf306
    # Source Nodes: [shortcut_28, x_461, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
    buf308 = extern_kernels.convolution(buf307, arg327_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf308, (8, 1104, 8, 8), (70656, 1, 8832, 1104))
    del arg327_1
    del buf307
    buf309 = buf308; del buf308  # reuse
    buf310 = buf309; del buf309  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_118(c_void_p(buf310.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()))
    del arg166_1
    del arg167_1
    del arg502_1
    del arg503_1
    # Source Nodes: [x_470, x_471], Original ATen: [aten.convolution, aten.hardswish]
    buf311 = extern_kernels.convolution(buf310, arg328_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
    assert_size_stride(buf311, (8, 1104, 8, 8), (70656, 1, 8832, 1104))
    del arg328_1
    del buf310
    buf312 = buf311; del buf311  # reuse
    buf313 = empty_strided((8, 1104, 1, 1), (1104, 1, 8832, 8832), device='cpu', dtype=torch.float32)
    buf314 = reinterpret_tensor(buf313, (8, 1104, 1, 1), (1104, 1, 1104, 1104), 0); del buf313  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_119(c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg504_1.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()))
    del arg168_1
    del arg169_1
    del arg504_1
    del arg505_1
    # Source Nodes: [x_475, x_se_68, x_se_69], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf315 = extern_kernels.convolution(buf314, arg329_1, arg330_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf315, (8, 48, 1, 1), (48, 1, 48, 48))
    del arg329_1
    del arg330_1
    del buf314
    buf316 = buf315; del buf315  # reuse
    cpp_fused_hardswish_120(c_void_p(buf316.data_ptr()))
    # Source Nodes: [x_se_70, x_se_71], Original ATen: [aten.convolution, aten.hardswish]
    buf317 = extern_kernels.convolution(buf316, arg331_1, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf317, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    del arg331_1
    del arg332_1
    del buf316
    buf318 = buf312; del buf312  # reuse
    cpp_fused_hardsigmoid_hardswish_mul_121(c_void_p(buf318.data_ptr()), c_void_p(buf317.data_ptr()))
    del buf317
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_475, x_476, x_477], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mul]
    buf319 = extern_kernels.convolution(buf318, arg333_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf319, (8, 224, 8, 8), (14336, 1, 1792, 224))
    del arg333_1
    del buf318
    buf320 = buf319; del buf319  # reuse
    cpp_fused__native_batch_norm_legit_no_training_122(c_void_p(buf320.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg507_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()))
    del arg170_1
    del arg171_1
    del arg506_1
    del arg507_1
    # Source Nodes: [x_478, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf321 = extern_kernels.convolution(buf320, arg334_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf321, (8, 1344, 8, 8), (86016, 1, 10752, 1344))
    del arg334_1
    del buf320
    buf322 = buf321; del buf321  # reuse
    buf323 = empty_strided((8, 1344, 1, 1), (1344, 1, 10752, 10752), device='cpu', dtype=torch.float32)
    buf324 = reinterpret_tensor(buf323, (8, 1344, 1, 1), (1344, 1, 1344, 1344), 0); del buf323  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_mean_123(c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(arg508_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()))
    del arg172_1
    del arg173_1
    del arg508_1
    del arg509_1
    del buf322
    # Source Nodes: [x_488, x_489, x_492], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
    buf325 = extern_kernels.convolution(buf324, arg335_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf325, (8, 1984, 1, 1), (1984, 1, 1984, 1984))
    del arg335_1
    del buf324
    buf326 = reinterpret_tensor(buf325, (8, 1984, 1, 1), (1984, 1, 1, 1), 0); del buf325  # reuse
    cpp_fused_hardswish_124(c_void_p(buf326.data_ptr()))
    buf327 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_495], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf326, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg174_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf327)
    del arg174_1
    del arg175_1
    return (buf327, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1000, 1984), (1984, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
