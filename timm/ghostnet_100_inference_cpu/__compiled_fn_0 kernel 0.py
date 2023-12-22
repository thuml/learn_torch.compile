
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(8);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (8L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(16);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-8L) + x1 + (8L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-8L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-8L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-8L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-8L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (16L*x0))] = tmp29;
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
    }
}
''')


cpp_fused_add_cat_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (16L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(8);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (8L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(16);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-8L) + x1 + (8L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-8L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-8L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-8L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-8L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (16L*x0))] = tmp30;
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
    }
}
''')


cpp_fused_cat_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(24);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (24L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(48);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-24L) + x1 + (24L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-24L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-24L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-24L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-24L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (48L*x0))] = tmp29;
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_9 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (12L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (12L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (12L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp14;
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_11 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (24L*x0))];
                    auto tmp30 = in_ptr6[static_cast<long>(x1)];
                    auto tmp32 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp42 = in_ptr9[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(12);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (12L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(24);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-12L) + x1 + (12L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-12L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-12L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-12L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-12L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = std::sqrt(tmp34);
                    auto tmp36 = 1 / tmp35;
                    auto tmp37 = static_cast<float>(1.0);
                    auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                    auto tmp39 = decltype(tmp31)(tmp31 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 * tmp40);
                    auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                    auto tmp44 = decltype(tmp28)(tmp28 + tmp43);
                    in_out_ptr0[static_cast<long>(x1 + (24L*x0))] = tmp44;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(36);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(72);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-36L) + x1 + (36L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-36L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-36L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-36L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-36L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (72L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (12L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (12L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (12L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (24L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(12);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (12L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(24);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-12L) + x1 + (12L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-12L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-12L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-12L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-12L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (24L*x0))] = tmp30;
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(36);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(72);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-36L) + x1 + (36L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-36L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-36L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-36L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-36L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (72L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_18 = async_compile.cpp('''
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_20 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (20L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (20L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (20L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (20L*x0))] = tmp14;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_23 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (40L*x0))];
                    auto tmp30 = in_ptr6[static_cast<long>(x1)];
                    auto tmp32 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp42 = in_ptr9[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-20L) + x1 + (20L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-20L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-20L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-20L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-20L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = std::sqrt(tmp34);
                    auto tmp36 = 1 / tmp35;
                    auto tmp37 = static_cast<float>(1.0);
                    auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                    auto tmp39 = decltype(tmp31)(tmp31 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 * tmp40);
                    auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                    auto tmp44 = decltype(tmp28)(tmp28 + tmp43);
                    in_out_ptr0[static_cast<long>(x1 + (40L*x0))] = tmp44;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (60L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (60L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_mean_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(60);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (60L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(120);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-60L) + x1 + (60L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-60L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-60L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-60L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-60L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (120L*x0))] = tmp29;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_26 = async_compile.cpp('''
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


cpp_fused_hardsigmoid_mul_27 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (20L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (20L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (20L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (20L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (40L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-20L) + x1 + (20L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-20L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-20L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-20L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-20L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (40L*x0))] = tmp30;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(120);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(240);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-120L) + x1 + (120L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-120L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-120L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-120L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-120L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (240L*x0))] = tmp29;
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_33 = async_compile.cpp('''
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
''')


cpp_fused__native_batch_norm_legit_no_training_34 = async_compile.cpp('''
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
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_35 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (80L*x0))];
                    auto tmp30 = in_ptr6[static_cast<long>(x1)];
                    auto tmp32 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp42 = in_ptr9[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-40L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-40L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = std::sqrt(tmp34);
                    auto tmp36 = 1 / tmp35;
                    auto tmp37 = static_cast<float>(1.0);
                    auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                    auto tmp39 = decltype(tmp31)(tmp31 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 * tmp40);
                    auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                    auto tmp44 = decltype(tmp28)(tmp28 + tmp43);
                    in_out_ptr0[static_cast<long>(x1 + (80L*x0))] = tmp44;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (100L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (100L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (100L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(100);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(200);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-100L) + x1 + (100L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-100L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-100L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-100L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-100L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (200L*x0))] = tmp29;
                }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_cat_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-40L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-40L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (80L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (92L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (92L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(92);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(184);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-92L) + x1 + (92L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-92L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-92L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-92L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-92L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (184L*x0))] = tmp29;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_cat_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-40L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-40L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (80L*x0))] = tmp30;
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (92L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (92L*x0))] = tmp15;
                }
            }
        }
    }
}
''')


cpp_fused_cat_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(92);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(184);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-92L) + x1 + (92L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-92L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-92L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-92L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-92L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (184L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_46 = async_compile.cpp('''
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
''')


cpp_fused_add_cat_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-40L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-40L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (80L*x0))] = tmp30;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_mean_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(240);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(480);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-240L) + x1 + (240L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-240L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-240L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-240L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-240L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp29;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_51 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_52 = async_compile.cpp('''
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_54 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (112L*x0))];
                    auto tmp30 = in_ptr6[static_cast<long>(x1)];
                    auto tmp32 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp42 = in_ptr9[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(56);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (56L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(112);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-56L) + x1 + (56L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-56L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-56L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-56L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-56L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = std::sqrt(tmp34);
                    auto tmp36 = 1 / tmp35;
                    auto tmp37 = static_cast<float>(1.0);
                    auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                    auto tmp39 = decltype(tmp31)(tmp31 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 * tmp40);
                    auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                    auto tmp44 = decltype(tmp28)(tmp28 + tmp43);
                    in_out_ptr0[static_cast<long>(x1 + (112L*x0))] = tmp44;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_mean_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(336);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (336L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(672);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-336L) + x1 + (336L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-336L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-336L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-336L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-336L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (672L*x0))] = tmp29;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_59 = async_compile.cpp('''
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_cat_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (112L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(56);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (56L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(112);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-56L) + x1 + (56L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-56L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-56L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-56L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-56L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp28 = tmp4 ? tmp7 : tmp27;
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (112L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_61 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(336);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (336L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(672);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-336L) + x1 + (336L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-336L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-336L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-336L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-336L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (672L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_65 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_66 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_67 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_68 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp30 = in_ptr6[static_cast<long>(x1)];
                auto tmp32 = in_ptr7[static_cast<long>(x1)];
                auto tmp40 = in_ptr8[static_cast<long>(x1)];
                auto tmp42 = in_ptr9[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = std::sqrt(tmp17);
                    auto tmp19 = 1 / tmp18;
                    auto tmp20 = static_cast<float>(1.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                    auto tmp23 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                    return tmp26;
                }
                ;
                auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp28 = tmp4 ? tmp7 : tmp27;
                auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                auto tmp33 = static_cast<float>(1e-05);
                auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                auto tmp35 = std::sqrt(tmp34);
                auto tmp36 = 1 / tmp35;
                auto tmp37 = static_cast<float>(1.0);
                auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                auto tmp39 = decltype(tmp31)(tmp31 * tmp38);
                auto tmp41 = decltype(tmp39)(tmp39 * tmp40);
                auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                auto tmp44 = decltype(tmp28)(tmp28 + tmp43);
                in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp44;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
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


cpp_fused_cat_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(480);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(960);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-480L) + x1 + (480L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-480L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-480L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-480L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-480L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (960L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_71 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_cat_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = std::sqrt(tmp17);
                    auto tmp19 = 1 / tmp18;
                    auto tmp20 = static_cast<float>(1.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                    auto tmp23 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                    return tmp26;
                }
                ;
                auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp28 = tmp4 ? tmp7 : tmp27;
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
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


cpp_fused_cat_mean_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(480);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(960);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-480L) + x1 + (480L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-480L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-480L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-480L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-480L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (960L*x0))] = tmp29;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_76 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_77 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_cat_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = std::sqrt(tmp17);
                    auto tmp19 = 1 / tmp18;
                    auto tmp20 = static_cast<float>(1.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                    auto tmp23 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                    return tmp26;
                }
                ;
                auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp28 = tmp4 ? tmp7 : tmp27;
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_79 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
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


cpp_fused_cat_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(480);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(960);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-480L) + x1 + (480L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-480L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-480L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-480L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-480L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (960L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_81 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_cat_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = std::sqrt(tmp17);
                    auto tmp19 = 1 / tmp18;
                    auto tmp20 = static_cast<float>(1.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                    auto tmp23 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                    return tmp26;
                }
                ;
                auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp28 = tmp4 ? tmp7 : tmp27;
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_83 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
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


cpp_fused_cat_mean_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(480);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(960);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-480L) + x1 + (480L*x0))];
                        auto tmp13 = in_ptr2[static_cast<long>((-480L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr3[static_cast<long>((-480L) + x1)];
                        auto tmp16 = static_cast<float>(1e-05);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = std::sqrt(tmp17);
                        auto tmp19 = 1 / tmp18;
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                        auto tmp23 = in_ptr4[static_cast<long>((-480L) + x1)];
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = in_ptr5[static_cast<long>((-480L) + x1)];
                        auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                        auto tmp27 = tmp26 * (tmp26>0);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp29 = tmp4 ? tmp7 : tmp28;
                    out_ptr0[static_cast<long>(x1 + (960L*x0))] = tmp29;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_86 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = static_cast<float>(3.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 + tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = at::vec::maximum(tmp4, tmp6);
                        auto tmp8 = static_cast<float>(6.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = at::vec::minimum(tmp7, tmp9);
                        auto tmp11 = tmp10 / tmp9;
                        auto tmp12 = tmp0 * tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_87 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_cat_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = std::sqrt(tmp17);
                    auto tmp19 = 1 / tmp18;
                    auto tmp20 = static_cast<float>(1.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp14)(tmp14 * tmp21);
                    auto tmp23 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                    return tmp26;
                }
                ;
                auto tmp27 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp28 = tmp4 ? tmp7 : tmp27;
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1 = args
    args.clear()
    assert_size_stride(arg0_1, (960, ), (1, ))
    assert_size_stride(arg1_1, (960, ), (1, ))
    assert_size_stride(arg2_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg3_1, (1000, ), (1, ))
    assert_size_stride(arg4_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg8_1, (8, ), (1, ))
    assert_size_stride(arg9_1, (8, ), (1, ))
    assert_size_stride(arg10_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg11_1, (8, ), (1, ))
    assert_size_stride(arg12_1, (8, ), (1, ))
    assert_size_stride(arg13_1, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg14_1, (8, ), (1, ))
    assert_size_stride(arg15_1, (8, ), (1, ))
    assert_size_stride(arg16_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg17_1, (8, ), (1, ))
    assert_size_stride(arg18_1, (8, ), (1, ))
    assert_size_stride(arg19_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg20_1, (24, ), (1, ))
    assert_size_stride(arg21_1, (24, ), (1, ))
    assert_size_stride(arg22_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg26_1, (48, ), (1, ))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg29_1, (12, ), (1, ))
    assert_size_stride(arg30_1, (12, ), (1, ))
    assert_size_stride(arg31_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg32_1, (12, ), (1, ))
    assert_size_stride(arg33_1, (12, ), (1, ))
    assert_size_stride(arg34_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg35_1, (16, ), (1, ))
    assert_size_stride(arg36_1, (16, ), (1, ))
    assert_size_stride(arg37_1, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg38_1, (24, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg41_1, (36, ), (1, ))
    assert_size_stride(arg42_1, (36, ), (1, ))
    assert_size_stride(arg43_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg44_1, (36, ), (1, ))
    assert_size_stride(arg45_1, (36, ), (1, ))
    assert_size_stride(arg46_1, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg47_1, (12, ), (1, ))
    assert_size_stride(arg48_1, (12, ), (1, ))
    assert_size_stride(arg49_1, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg50_1, (12, ), (1, ))
    assert_size_stride(arg51_1, (12, ), (1, ))
    assert_size_stride(arg52_1, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg53_1, (36, ), (1, ))
    assert_size_stride(arg54_1, (36, ), (1, ))
    assert_size_stride(arg55_1, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg56_1, (36, ), (1, ))
    assert_size_stride(arg57_1, (36, ), (1, ))
    assert_size_stride(arg58_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg59_1, (72, ), (1, ))
    assert_size_stride(arg60_1, (72, ), (1, ))
    assert_size_stride(arg61_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg62_1, (20, ), (1, ))
    assert_size_stride(arg63_1, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg64_1, (72, ), (1, ))
    assert_size_stride(arg65_1, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg66_1, (20, ), (1, ))
    assert_size_stride(arg67_1, (20, ), (1, ))
    assert_size_stride(arg68_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg69_1, (20, ), (1, ))
    assert_size_stride(arg70_1, (20, ), (1, ))
    assert_size_stride(arg71_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg72_1, (24, ), (1, ))
    assert_size_stride(arg73_1, (24, ), (1, ))
    assert_size_stride(arg74_1, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg75_1, (40, ), (1, ))
    assert_size_stride(arg76_1, (40, ), (1, ))
    assert_size_stride(arg77_1, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg78_1, (60, ), (1, ))
    assert_size_stride(arg79_1, (60, ), (1, ))
    assert_size_stride(arg80_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg81_1, (60, ), (1, ))
    assert_size_stride(arg82_1, (60, ), (1, ))
    assert_size_stride(arg83_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg84_1, (32, ), (1, ))
    assert_size_stride(arg85_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg86_1, (120, ), (1, ))
    assert_size_stride(arg87_1, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg88_1, (20, ), (1, ))
    assert_size_stride(arg89_1, (20, ), (1, ))
    assert_size_stride(arg90_1, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg91_1, (20, ), (1, ))
    assert_size_stride(arg92_1, (20, ), (1, ))
    assert_size_stride(arg93_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg94_1, (120, ), (1, ))
    assert_size_stride(arg95_1, (120, ), (1, ))
    assert_size_stride(arg96_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (120, ), (1, ))
    assert_size_stride(arg98_1, (120, ), (1, ))
    assert_size_stride(arg99_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (240, ), (1, ))
    assert_size_stride(arg101_1, (240, ), (1, ))
    assert_size_stride(arg102_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg103_1, (40, ), (1, ))
    assert_size_stride(arg104_1, (40, ), (1, ))
    assert_size_stride(arg105_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (40, ), (1, ))
    assert_size_stride(arg107_1, (40, ), (1, ))
    assert_size_stride(arg108_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (40, ), (1, ))
    assert_size_stride(arg110_1, (40, ), (1, ))
    assert_size_stride(arg111_1, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg112_1, (80, ), (1, ))
    assert_size_stride(arg113_1, (80, ), (1, ))
    assert_size_stride(arg114_1, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg115_1, (100, ), (1, ))
    assert_size_stride(arg116_1, (100, ), (1, ))
    assert_size_stride(arg117_1, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (100, ), (1, ))
    assert_size_stride(arg119_1, (100, ), (1, ))
    assert_size_stride(arg120_1, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg121_1, (40, ), (1, ))
    assert_size_stride(arg122_1, (40, ), (1, ))
    assert_size_stride(arg123_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (40, ), (1, ))
    assert_size_stride(arg125_1, (40, ), (1, ))
    assert_size_stride(arg126_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg127_1, (92, ), (1, ))
    assert_size_stride(arg128_1, (92, ), (1, ))
    assert_size_stride(arg129_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg130_1, (92, ), (1, ))
    assert_size_stride(arg131_1, (92, ), (1, ))
    assert_size_stride(arg132_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg133_1, (40, ), (1, ))
    assert_size_stride(arg134_1, (40, ), (1, ))
    assert_size_stride(arg135_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (40, ), (1, ))
    assert_size_stride(arg137_1, (40, ), (1, ))
    assert_size_stride(arg138_1, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg139_1, (92, ), (1, ))
    assert_size_stride(arg140_1, (92, ), (1, ))
    assert_size_stride(arg141_1, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (92, ), (1, ))
    assert_size_stride(arg143_1, (92, ), (1, ))
    assert_size_stride(arg144_1, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg145_1, (40, ), (1, ))
    assert_size_stride(arg146_1, (40, ), (1, ))
    assert_size_stride(arg147_1, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (40, ), (1, ))
    assert_size_stride(arg149_1, (40, ), (1, ))
    assert_size_stride(arg150_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg151_1, (240, ), (1, ))
    assert_size_stride(arg152_1, (240, ), (1, ))
    assert_size_stride(arg153_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (240, ), (1, ))
    assert_size_stride(arg155_1, (240, ), (1, ))
    assert_size_stride(arg156_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg157_1, (120, ), (1, ))
    assert_size_stride(arg158_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg161_1, (56, ), (1, ))
    assert_size_stride(arg162_1, (56, ), (1, ))
    assert_size_stride(arg163_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg164_1, (56, ), (1, ))
    assert_size_stride(arg165_1, (56, ), (1, ))
    assert_size_stride(arg166_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (80, ), (1, ))
    assert_size_stride(arg168_1, (80, ), (1, ))
    assert_size_stride(arg169_1, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg170_1, (112, ), (1, ))
    assert_size_stride(arg171_1, (112, ), (1, ))
    assert_size_stride(arg172_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg173_1, (336, ), (1, ))
    assert_size_stride(arg174_1, (336, ), (1, ))
    assert_size_stride(arg175_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg176_1, (336, ), (1, ))
    assert_size_stride(arg177_1, (336, ), (1, ))
    assert_size_stride(arg178_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg179_1, (168, ), (1, ))
    assert_size_stride(arg180_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg181_1, (672, ), (1, ))
    assert_size_stride(arg182_1, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg183_1, (56, ), (1, ))
    assert_size_stride(arg184_1, (56, ), (1, ))
    assert_size_stride(arg185_1, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg186_1, (56, ), (1, ))
    assert_size_stride(arg187_1, (56, ), (1, ))
    assert_size_stride(arg188_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg189_1, (336, ), (1, ))
    assert_size_stride(arg190_1, (336, ), (1, ))
    assert_size_stride(arg191_1, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg192_1, (336, ), (1, ))
    assert_size_stride(arg193_1, (336, ), (1, ))
    assert_size_stride(arg194_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (672, ), (1, ))
    assert_size_stride(arg197_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg198_1, (168, ), (1, ))
    assert_size_stride(arg199_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg200_1, (672, ), (1, ))
    assert_size_stride(arg201_1, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg202_1, (80, ), (1, ))
    assert_size_stride(arg203_1, (80, ), (1, ))
    assert_size_stride(arg204_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg205_1, (80, ), (1, ))
    assert_size_stride(arg206_1, (80, ), (1, ))
    assert_size_stride(arg207_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg208_1, (112, ), (1, ))
    assert_size_stride(arg209_1, (112, ), (1, ))
    assert_size_stride(arg210_1, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg211_1, (160, ), (1, ))
    assert_size_stride(arg212_1, (160, ), (1, ))
    assert_size_stride(arg213_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg214_1, (480, ), (1, ))
    assert_size_stride(arg215_1, (480, ), (1, ))
    assert_size_stride(arg216_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (480, ), (1, ))
    assert_size_stride(arg218_1, (480, ), (1, ))
    assert_size_stride(arg219_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg220_1, (80, ), (1, ))
    assert_size_stride(arg221_1, (80, ), (1, ))
    assert_size_stride(arg222_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg223_1, (80, ), (1, ))
    assert_size_stride(arg224_1, (80, ), (1, ))
    assert_size_stride(arg225_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg226_1, (480, ), (1, ))
    assert_size_stride(arg227_1, (480, ), (1, ))
    assert_size_stride(arg228_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg229_1, (480, ), (1, ))
    assert_size_stride(arg230_1, (480, ), (1, ))
    assert_size_stride(arg231_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg232_1, (240, ), (1, ))
    assert_size_stride(arg233_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg234_1, (960, ), (1, ))
    assert_size_stride(arg235_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg236_1, (80, ), (1, ))
    assert_size_stride(arg237_1, (80, ), (1, ))
    assert_size_stride(arg238_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg239_1, (80, ), (1, ))
    assert_size_stride(arg240_1, (80, ), (1, ))
    assert_size_stride(arg241_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg242_1, (480, ), (1, ))
    assert_size_stride(arg243_1, (480, ), (1, ))
    assert_size_stride(arg244_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg245_1, (480, ), (1, ))
    assert_size_stride(arg246_1, (480, ), (1, ))
    assert_size_stride(arg247_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg248_1, (80, ), (1, ))
    assert_size_stride(arg249_1, (80, ), (1, ))
    assert_size_stride(arg250_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg251_1, (80, ), (1, ))
    assert_size_stride(arg252_1, (80, ), (1, ))
    assert_size_stride(arg253_1, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg254_1, (480, ), (1, ))
    assert_size_stride(arg255_1, (480, ), (1, ))
    assert_size_stride(arg256_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg257_1, (480, ), (1, ))
    assert_size_stride(arg258_1, (480, ), (1, ))
    assert_size_stride(arg259_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg260_1, (240, ), (1, ))
    assert_size_stride(arg261_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg262_1, (960, ), (1, ))
    assert_size_stride(arg263_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg264_1, (80, ), (1, ))
    assert_size_stride(arg265_1, (80, ), (1, ))
    assert_size_stride(arg266_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg267_1, (80, ), (1, ))
    assert_size_stride(arg268_1, (80, ), (1, ))
    assert_size_stride(arg269_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg270_1, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg271_1, (1280, ), (1, ))
    assert_size_stride(arg272_1, (960, ), (1, ))
    assert_size_stride(arg273_1, (960, ), (1, ))
    assert_size_stride(arg274_1, (16, ), (1, ))
    assert_size_stride(arg275_1, (16, ), (1, ))
    assert_size_stride(arg276_1, (), ())
    assert_size_stride(arg277_1, (8, ), (1, ))
    assert_size_stride(arg278_1, (8, ), (1, ))
    assert_size_stride(arg279_1, (), ())
    assert_size_stride(arg280_1, (8, ), (1, ))
    assert_size_stride(arg281_1, (8, ), (1, ))
    assert_size_stride(arg282_1, (), ())
    assert_size_stride(arg283_1, (8, ), (1, ))
    assert_size_stride(arg284_1, (8, ), (1, ))
    assert_size_stride(arg285_1, (), ())
    assert_size_stride(arg286_1, (8, ), (1, ))
    assert_size_stride(arg287_1, (8, ), (1, ))
    assert_size_stride(arg288_1, (), ())
    assert_size_stride(arg289_1, (24, ), (1, ))
    assert_size_stride(arg290_1, (24, ), (1, ))
    assert_size_stride(arg291_1, (), ())
    assert_size_stride(arg292_1, (24, ), (1, ))
    assert_size_stride(arg293_1, (24, ), (1, ))
    assert_size_stride(arg294_1, (), ())
    assert_size_stride(arg295_1, (48, ), (1, ))
    assert_size_stride(arg296_1, (48, ), (1, ))
    assert_size_stride(arg297_1, (), ())
    assert_size_stride(arg298_1, (12, ), (1, ))
    assert_size_stride(arg299_1, (12, ), (1, ))
    assert_size_stride(arg300_1, (), ())
    assert_size_stride(arg301_1, (12, ), (1, ))
    assert_size_stride(arg302_1, (12, ), (1, ))
    assert_size_stride(arg303_1, (), ())
    assert_size_stride(arg304_1, (16, ), (1, ))
    assert_size_stride(arg305_1, (16, ), (1, ))
    assert_size_stride(arg306_1, (), ())
    assert_size_stride(arg307_1, (24, ), (1, ))
    assert_size_stride(arg308_1, (24, ), (1, ))
    assert_size_stride(arg309_1, (), ())
    assert_size_stride(arg310_1, (36, ), (1, ))
    assert_size_stride(arg311_1, (36, ), (1, ))
    assert_size_stride(arg312_1, (), ())
    assert_size_stride(arg313_1, (36, ), (1, ))
    assert_size_stride(arg314_1, (36, ), (1, ))
    assert_size_stride(arg315_1, (), ())
    assert_size_stride(arg316_1, (12, ), (1, ))
    assert_size_stride(arg317_1, (12, ), (1, ))
    assert_size_stride(arg318_1, (), ())
    assert_size_stride(arg319_1, (12, ), (1, ))
    assert_size_stride(arg320_1, (12, ), (1, ))
    assert_size_stride(arg321_1, (), ())
    assert_size_stride(arg322_1, (36, ), (1, ))
    assert_size_stride(arg323_1, (36, ), (1, ))
    assert_size_stride(arg324_1, (), ())
    assert_size_stride(arg325_1, (36, ), (1, ))
    assert_size_stride(arg326_1, (36, ), (1, ))
    assert_size_stride(arg327_1, (), ())
    assert_size_stride(arg328_1, (72, ), (1, ))
    assert_size_stride(arg329_1, (72, ), (1, ))
    assert_size_stride(arg330_1, (), ())
    assert_size_stride(arg331_1, (20, ), (1, ))
    assert_size_stride(arg332_1, (20, ), (1, ))
    assert_size_stride(arg333_1, (), ())
    assert_size_stride(arg334_1, (20, ), (1, ))
    assert_size_stride(arg335_1, (20, ), (1, ))
    assert_size_stride(arg336_1, (), ())
    assert_size_stride(arg337_1, (24, ), (1, ))
    assert_size_stride(arg338_1, (24, ), (1, ))
    assert_size_stride(arg339_1, (), ())
    assert_size_stride(arg340_1, (40, ), (1, ))
    assert_size_stride(arg341_1, (40, ), (1, ))
    assert_size_stride(arg342_1, (), ())
    assert_size_stride(arg343_1, (60, ), (1, ))
    assert_size_stride(arg344_1, (60, ), (1, ))
    assert_size_stride(arg345_1, (), ())
    assert_size_stride(arg346_1, (60, ), (1, ))
    assert_size_stride(arg347_1, (60, ), (1, ))
    assert_size_stride(arg348_1, (), ())
    assert_size_stride(arg349_1, (20, ), (1, ))
    assert_size_stride(arg350_1, (20, ), (1, ))
    assert_size_stride(arg351_1, (), ())
    assert_size_stride(arg352_1, (20, ), (1, ))
    assert_size_stride(arg353_1, (20, ), (1, ))
    assert_size_stride(arg354_1, (), ())
    assert_size_stride(arg355_1, (120, ), (1, ))
    assert_size_stride(arg356_1, (120, ), (1, ))
    assert_size_stride(arg357_1, (), ())
    assert_size_stride(arg358_1, (120, ), (1, ))
    assert_size_stride(arg359_1, (120, ), (1, ))
    assert_size_stride(arg360_1, (), ())
    assert_size_stride(arg361_1, (240, ), (1, ))
    assert_size_stride(arg362_1, (240, ), (1, ))
    assert_size_stride(arg363_1, (), ())
    assert_size_stride(arg364_1, (40, ), (1, ))
    assert_size_stride(arg365_1, (40, ), (1, ))
    assert_size_stride(arg366_1, (), ())
    assert_size_stride(arg367_1, (40, ), (1, ))
    assert_size_stride(arg368_1, (40, ), (1, ))
    assert_size_stride(arg369_1, (), ())
    assert_size_stride(arg370_1, (40, ), (1, ))
    assert_size_stride(arg371_1, (40, ), (1, ))
    assert_size_stride(arg372_1, (), ())
    assert_size_stride(arg373_1, (80, ), (1, ))
    assert_size_stride(arg374_1, (80, ), (1, ))
    assert_size_stride(arg375_1, (), ())
    assert_size_stride(arg376_1, (100, ), (1, ))
    assert_size_stride(arg377_1, (100, ), (1, ))
    assert_size_stride(arg378_1, (), ())
    assert_size_stride(arg379_1, (100, ), (1, ))
    assert_size_stride(arg380_1, (100, ), (1, ))
    assert_size_stride(arg381_1, (), ())
    assert_size_stride(arg382_1, (40, ), (1, ))
    assert_size_stride(arg383_1, (40, ), (1, ))
    assert_size_stride(arg384_1, (), ())
    assert_size_stride(arg385_1, (40, ), (1, ))
    assert_size_stride(arg386_1, (40, ), (1, ))
    assert_size_stride(arg387_1, (), ())
    assert_size_stride(arg388_1, (92, ), (1, ))
    assert_size_stride(arg389_1, (92, ), (1, ))
    assert_size_stride(arg390_1, (), ())
    assert_size_stride(arg391_1, (92, ), (1, ))
    assert_size_stride(arg392_1, (92, ), (1, ))
    assert_size_stride(arg393_1, (), ())
    assert_size_stride(arg394_1, (40, ), (1, ))
    assert_size_stride(arg395_1, (40, ), (1, ))
    assert_size_stride(arg396_1, (), ())
    assert_size_stride(arg397_1, (40, ), (1, ))
    assert_size_stride(arg398_1, (40, ), (1, ))
    assert_size_stride(arg399_1, (), ())
    assert_size_stride(arg400_1, (92, ), (1, ))
    assert_size_stride(arg401_1, (92, ), (1, ))
    assert_size_stride(arg402_1, (), ())
    assert_size_stride(arg403_1, (92, ), (1, ))
    assert_size_stride(arg404_1, (92, ), (1, ))
    assert_size_stride(arg405_1, (), ())
    assert_size_stride(arg406_1, (40, ), (1, ))
    assert_size_stride(arg407_1, (40, ), (1, ))
    assert_size_stride(arg408_1, (), ())
    assert_size_stride(arg409_1, (40, ), (1, ))
    assert_size_stride(arg410_1, (40, ), (1, ))
    assert_size_stride(arg411_1, (), ())
    assert_size_stride(arg412_1, (240, ), (1, ))
    assert_size_stride(arg413_1, (240, ), (1, ))
    assert_size_stride(arg414_1, (), ())
    assert_size_stride(arg415_1, (240, ), (1, ))
    assert_size_stride(arg416_1, (240, ), (1, ))
    assert_size_stride(arg417_1, (), ())
    assert_size_stride(arg418_1, (56, ), (1, ))
    assert_size_stride(arg419_1, (56, ), (1, ))
    assert_size_stride(arg420_1, (), ())
    assert_size_stride(arg421_1, (56, ), (1, ))
    assert_size_stride(arg422_1, (56, ), (1, ))
    assert_size_stride(arg423_1, (), ())
    assert_size_stride(arg424_1, (80, ), (1, ))
    assert_size_stride(arg425_1, (80, ), (1, ))
    assert_size_stride(arg426_1, (), ())
    assert_size_stride(arg427_1, (112, ), (1, ))
    assert_size_stride(arg428_1, (112, ), (1, ))
    assert_size_stride(arg429_1, (), ())
    assert_size_stride(arg430_1, (336, ), (1, ))
    assert_size_stride(arg431_1, (336, ), (1, ))
    assert_size_stride(arg432_1, (), ())
    assert_size_stride(arg433_1, (336, ), (1, ))
    assert_size_stride(arg434_1, (336, ), (1, ))
    assert_size_stride(arg435_1, (), ())
    assert_size_stride(arg436_1, (56, ), (1, ))
    assert_size_stride(arg437_1, (56, ), (1, ))
    assert_size_stride(arg438_1, (), ())
    assert_size_stride(arg439_1, (56, ), (1, ))
    assert_size_stride(arg440_1, (56, ), (1, ))
    assert_size_stride(arg441_1, (), ())
    assert_size_stride(arg442_1, (336, ), (1, ))
    assert_size_stride(arg443_1, (336, ), (1, ))
    assert_size_stride(arg444_1, (), ())
    assert_size_stride(arg445_1, (336, ), (1, ))
    assert_size_stride(arg446_1, (336, ), (1, ))
    assert_size_stride(arg447_1, (), ())
    assert_size_stride(arg448_1, (672, ), (1, ))
    assert_size_stride(arg449_1, (672, ), (1, ))
    assert_size_stride(arg450_1, (), ())
    assert_size_stride(arg451_1, (80, ), (1, ))
    assert_size_stride(arg452_1, (80, ), (1, ))
    assert_size_stride(arg453_1, (), ())
    assert_size_stride(arg454_1, (80, ), (1, ))
    assert_size_stride(arg455_1, (80, ), (1, ))
    assert_size_stride(arg456_1, (), ())
    assert_size_stride(arg457_1, (112, ), (1, ))
    assert_size_stride(arg458_1, (112, ), (1, ))
    assert_size_stride(arg459_1, (), ())
    assert_size_stride(arg460_1, (160, ), (1, ))
    assert_size_stride(arg461_1, (160, ), (1, ))
    assert_size_stride(arg462_1, (), ())
    assert_size_stride(arg463_1, (480, ), (1, ))
    assert_size_stride(arg464_1, (480, ), (1, ))
    assert_size_stride(arg465_1, (), ())
    assert_size_stride(arg466_1, (480, ), (1, ))
    assert_size_stride(arg467_1, (480, ), (1, ))
    assert_size_stride(arg468_1, (), ())
    assert_size_stride(arg469_1, (80, ), (1, ))
    assert_size_stride(arg470_1, (80, ), (1, ))
    assert_size_stride(arg471_1, (), ())
    assert_size_stride(arg472_1, (80, ), (1, ))
    assert_size_stride(arg473_1, (80, ), (1, ))
    assert_size_stride(arg474_1, (), ())
    assert_size_stride(arg475_1, (480, ), (1, ))
    assert_size_stride(arg476_1, (480, ), (1, ))
    assert_size_stride(arg477_1, (), ())
    assert_size_stride(arg478_1, (480, ), (1, ))
    assert_size_stride(arg479_1, (480, ), (1, ))
    assert_size_stride(arg480_1, (), ())
    assert_size_stride(arg481_1, (80, ), (1, ))
    assert_size_stride(arg482_1, (80, ), (1, ))
    assert_size_stride(arg483_1, (), ())
    assert_size_stride(arg484_1, (80, ), (1, ))
    assert_size_stride(arg485_1, (80, ), (1, ))
    assert_size_stride(arg486_1, (), ())
    assert_size_stride(arg487_1, (480, ), (1, ))
    assert_size_stride(arg488_1, (480, ), (1, ))
    assert_size_stride(arg489_1, (), ())
    assert_size_stride(arg490_1, (480, ), (1, ))
    assert_size_stride(arg491_1, (480, ), (1, ))
    assert_size_stride(arg492_1, (), ())
    assert_size_stride(arg493_1, (80, ), (1, ))
    assert_size_stride(arg494_1, (80, ), (1, ))
    assert_size_stride(arg495_1, (), ())
    assert_size_stride(arg496_1, (80, ), (1, ))
    assert_size_stride(arg497_1, (80, ), (1, ))
    assert_size_stride(arg498_1, (), ())
    assert_size_stride(arg499_1, (480, ), (1, ))
    assert_size_stride(arg500_1, (480, ), (1, ))
    assert_size_stride(arg501_1, (), ())
    assert_size_stride(arg502_1, (480, ), (1, ))
    assert_size_stride(arg503_1, (480, ), (1, ))
    assert_size_stride(arg504_1, (), ())
    assert_size_stride(arg505_1, (80, ), (1, ))
    assert_size_stride(arg506_1, (80, ), (1, ))
    assert_size_stride(arg507_1, (), ())
    assert_size_stride(arg508_1, (80, ), (1, ))
    assert_size_stride(arg509_1, (80, ), (1, ))
    assert_size_stride(arg510_1, (), ())
    assert_size_stride(arg511_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg511_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg4_1
    del arg511_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()))
    del arg274_1
    del arg275_1
    del arg5_1
    del arg6_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg7_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf4, (8, 8, 112, 112), (100352, 1, 896, 8))
    del arg7_1
    buf5 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg277_1
    del arg278_1
    del arg8_1
    del arg9_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf6 = extern_kernels.convolution(buf5, arg10_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf6, (8, 8, 112, 112), (100352, 1, 896, 8))
    del arg10_1
    buf7 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_3(c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg11_1
    del arg12_1
    del arg280_1
    del arg281_1
    del buf5
    del buf6
    # Source Nodes: [cat_63, getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf8 = extern_kernels.convolution(buf7, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 8, 112, 112), (100352, 1, 896, 8))
    del arg13_1
    del buf7
    buf9 = buf8; del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_4(c_void_p(buf9.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg283_1
    del arg284_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf10, (8, 8, 112, 112), (100352, 1, 896, 8))
    del arg16_1
    buf11 = buf3; del buf3  # reuse
    cpp_fused_add_cat_5(c_void_p(buf11.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()))
    del arg17_1
    del arg18_1
    del arg286_1
    del arg287_1
    del buf10
    del buf9
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, arg19_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (8, 24, 112, 112), (301056, 1, 2688, 24))
    del arg19_1
    buf13 = buf12; del buf12  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf13.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg289_1
    del arg290_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, arg22_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf14, (8, 24, 112, 112), (301056, 1, 2688, 24))
    del arg22_1
    buf15 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    cpp_fused_cat_7(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg23_1
    del arg24_1
    del arg292_1
    del arg293_1
    del buf13
    del buf14
    # Source Nodes: [cat_61, x_7], Original ATen: [aten.cat, aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg25_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf16, (8, 48, 56, 56), (150528, 1, 2688, 48))
    del arg25_1
    del buf15
    buf17 = buf16; del buf16  # reuse
    cpp_fused__native_batch_norm_legit_no_training_8(c_void_p(buf17.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    del arg295_1
    del arg296_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf18 = extern_kernels.convolution(buf17, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (8, 12, 56, 56), (37632, 1, 672, 12))
    del arg28_1
    del buf17
    buf19 = buf18; del buf18  # reuse
    cpp_fused__native_batch_norm_legit_no_training_9(c_void_p(buf19.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()))
    del arg298_1
    del arg299_1
    del arg29_1
    del arg30_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf19, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
    assert_size_stride(buf20, (8, 12, 56, 56), (37632, 1, 672, 12))
    del arg31_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf11, arg34_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf21, (8, 16, 56, 56), (50176, 1, 896, 16))
    del arg34_1
    del buf11
    buf22 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_10(c_void_p(buf22.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()))
    del arg304_1
    del arg305_1
    del arg35_1
    del arg36_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1, getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf23 = extern_kernels.convolution(buf22, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg37_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_11(c_void_p(buf24.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg301_1
    del arg302_1
    del arg307_1
    del arg308_1
    del arg32_1
    del arg33_1
    del arg38_1
    del arg39_1
    del buf19
    del buf20
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf25 = extern_kernels.convolution(buf24, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf25, (8, 36, 56, 56), (112896, 1, 2016, 36))
    del arg40_1
    buf26 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf26.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg42_1.data_ptr()))
    del arg310_1
    del arg311_1
    del arg41_1
    del arg42_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf26, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
    assert_size_stride(buf27, (8, 36, 56, 56), (112896, 1, 2016, 36))
    del arg43_1
    buf28 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused_cat_13(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg313_1
    del arg314_1
    del arg44_1
    del arg45_1
    del buf26
    del buf27
    # Source Nodes: [cat_59, getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf29 = extern_kernels.convolution(buf28, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 12, 56, 56), (37632, 1, 672, 12))
    del arg46_1
    buf30 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_14(c_void_p(buf30.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()))
    del arg316_1
    del arg317_1
    del arg47_1
    del arg48_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
    assert_size_stride(buf31, (8, 12, 56, 56), (37632, 1, 672, 12))
    del arg49_1
    buf32 = buf24; del buf24  # reuse
    cpp_fused_add_cat_15(c_void_p(buf32.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()))
    del arg319_1
    del arg320_1
    del arg50_1
    del arg51_1
    del buf30
    del buf31
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (8, 36, 56, 56), (112896, 1, 2016, 36))
    del arg52_1
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_16(c_void_p(buf34.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg53_1
    del arg54_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf35 = extern_kernels.convolution(buf34, arg55_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
    assert_size_stride(buf35, (8, 36, 56, 56), (112896, 1, 2016, 36))
    del arg55_1
    buf36 = buf28; del buf28  # reuse
    cpp_fused_cat_17(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg325_1
    del arg326_1
    del arg56_1
    del arg57_1
    del buf34
    del buf35
    # Source Nodes: [cat_57, x_15], Original ATen: [aten.cat, aten.convolution]
    buf37 = extern_kernels.convolution(buf36, arg58_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf37, (8, 72, 28, 28), (56448, 1, 2016, 72))
    del arg58_1
    del buf36
    buf38 = buf37; del buf37  # reuse
    buf39 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf40 = reinterpret_tensor(buf39, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_18(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()))
    del arg328_1
    del arg329_1
    del arg59_1
    del arg60_1
    # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
    buf41 = extern_kernels.convolution(buf40, arg61_1, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg61_1
    del arg62_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    cpp_fused_relu_19(c_void_p(buf42.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.relu]
    buf43 = extern_kernels.convolution(buf42, arg63_1, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf43, (8, 72, 1, 1), (72, 1, 72, 72))
    del arg63_1
    del arg64_1
    del buf42
    buf44 = buf38; del buf38  # reuse
    cpp_fused_hardsigmoid_mul_20(c_void_p(buf44.data_ptr()), c_void_p(buf43.data_ptr()))
    del buf43
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___3_____0___se_gate, x_17], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf45 = extern_kernels.convolution(buf44, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (8, 20, 28, 28), (15680, 1, 560, 20))
    del arg65_1
    del buf44
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_21(c_void_p(buf46.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg331_1
    del arg332_1
    del arg66_1
    del arg67_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf47 = extern_kernels.convolution(buf46, arg68_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
    assert_size_stride(buf47, (8, 20, 28, 28), (15680, 1, 560, 20))
    del arg68_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf32, arg71_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf48, (8, 24, 28, 28), (18816, 1, 672, 24))
    del arg71_1
    del buf32
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_22(c_void_p(buf49.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg337_1
    del arg338_1
    del arg72_1
    del arg73_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1, getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf50 = extern_kernels.convolution(buf49, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (8, 40, 28, 28), (31360, 1, 1120, 40))
    del arg74_1
    del buf49
    buf51 = buf50; del buf50  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_23(c_void_p(buf51.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()))
    del arg334_1
    del arg335_1
    del arg340_1
    del arg341_1
    del arg69_1
    del arg70_1
    del arg75_1
    del arg76_1
    del buf46
    del buf47
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf51, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg77_1
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf53.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg343_1
    del arg344_1
    del arg78_1
    del arg79_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf53, arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf54, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg80_1
    buf55 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf56, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf56  # reuse
    cpp_fused_cat_mean_25(c_void_p(buf57.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg346_1
    del arg347_1
    del arg81_1
    del arg82_1
    del buf53
    del buf54
    # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
    buf58 = extern_kernels.convolution(buf57, arg83_1, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (8, 32, 1, 1), (32, 1, 32, 32))
    del arg83_1
    del arg84_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    cpp_fused_relu_26(c_void_p(buf59.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.relu]
    buf60 = extern_kernels.convolution(buf59, arg85_1, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf60, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg85_1
    del arg86_1
    del buf59
    buf61 = buf55; del buf55  # reuse
    cpp_fused_hardsigmoid_mul_27(c_void_p(buf61.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf60
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___4_____0___se_gate, x_21], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf62 = extern_kernels.convolution(buf61, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 20, 28, 28), (15680, 1, 560, 20))
    del arg87_1
    del buf61
    buf63 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_28(c_void_p(buf63.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg349_1
    del arg350_1
    del arg88_1
    del arg89_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
    assert_size_stride(buf64, (8, 20, 28, 28), (15680, 1, 560, 20))
    del arg90_1
    buf65 = buf51; del buf51  # reuse
    cpp_fused_add_cat_29(c_void_p(buf65.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()))
    del arg352_1
    del arg353_1
    del arg91_1
    del arg92_1
    del buf63
    del buf64
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (8, 120, 28, 28), (94080, 1, 3360, 120))
    del arg93_1
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf67.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg355_1
    del arg356_1
    del arg94_1
    del arg95_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf68, (8, 120, 28, 28), (94080, 1, 3360, 120))
    del arg96_1
    buf69 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    cpp_fused_cat_31(c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg358_1
    del arg359_1
    del arg97_1
    del arg98_1
    del buf67
    # Source Nodes: [cat_53, x_25], Original ATen: [aten.cat, aten.convolution]
    buf70 = extern_kernels.convolution(buf69, arg99_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf70, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg99_1
    del buf69
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_32(c_void_p(buf71.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg361_1
    del arg362_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf72 = extern_kernels.convolution(buf71, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf72, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg102_1
    del buf71
    buf73 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_33(c_void_p(buf73.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()))
    del arg103_1
    del arg104_1
    del arg364_1
    del arg365_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf74, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg105_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_0], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf65, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf75, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg108_1
    del buf65
    buf76 = buf75; del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_34(c_void_p(buf76.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg370_1
    del arg371_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1, getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf77 = extern_kernels.convolution(buf76, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg111_1
    del buf76
    buf78 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_35(c_void_p(buf78.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg112_1
    del arg113_1
    del arg367_1
    del arg368_1
    del arg373_1
    del arg374_1
    del buf73
    del buf74
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf79 = extern_kernels.convolution(buf78, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (8, 100, 14, 14), (19600, 1, 1400, 100))
    del arg114_1
    buf80 = buf79; del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf80.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg376_1
    del arg377_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=100, bias=None)
    assert_size_stride(buf81, (8, 100, 14, 14), (19600, 1, 1400, 100))
    del arg117_1
    buf82 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    cpp_fused_cat_37(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg118_1
    del arg119_1
    del arg379_1
    del arg380_1
    del buf80
    del buf81
    # Source Nodes: [cat_51, getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf83 = extern_kernels.convolution(buf82, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf83, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg120_1
    del buf82
    buf84 = buf83; del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_38(c_void_p(buf84.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()))
    del arg121_1
    del arg122_1
    del arg382_1
    del arg383_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(buf84, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf85, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg123_1
    buf86 = buf78; del buf78  # reuse
    cpp_fused_add_cat_39(c_void_p(buf86.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    del arg385_1
    del arg386_1
    del buf84
    del buf85
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf87 = extern_kernels.convolution(buf86, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (8, 92, 14, 14), (18032, 1, 1288, 92))
    del arg126_1
    buf88 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_40(c_void_p(buf88.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg388_1
    del arg389_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(buf88, arg129_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
    assert_size_stride(buf89, (8, 92, 14, 14), (18032, 1, 1288, 92))
    del arg129_1
    buf90 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused_cat_41(c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg130_1
    del arg131_1
    del arg391_1
    del arg392_1
    del buf88
    del buf89
    # Source Nodes: [cat_49, getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf91 = extern_kernels.convolution(buf90, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg132_1
    buf92 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_42(c_void_p(buf92.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()))
    del arg133_1
    del arg134_1
    del arg394_1
    del arg395_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf93, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg135_1
    buf94 = buf86; del buf86  # reuse
    cpp_fused_add_cat_43(c_void_p(buf94.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg136_1
    del arg137_1
    del arg397_1
    del arg398_1
    del buf92
    del buf93
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 92, 14, 14), (18032, 1, 1288, 92))
    del arg138_1
    buf96 = buf95; del buf95  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf96.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()))
    del arg139_1
    del arg140_1
    del arg400_1
    del arg401_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(buf96, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
    assert_size_stride(buf97, (8, 92, 14, 14), (18032, 1, 1288, 92))
    del arg141_1
    buf98 = buf90; del buf90  # reuse
    cpp_fused_cat_45(c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg142_1
    del arg143_1
    del arg403_1
    del arg404_1
    del buf96
    del buf97
    # Source Nodes: [cat_47, getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf99 = extern_kernels.convolution(buf98, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg144_1
    del buf98
    buf100 = buf99; del buf99  # reuse
    cpp_fused__native_batch_norm_legit_no_training_46(c_void_p(buf100.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()))
    del arg145_1
    del arg146_1
    del arg406_1
    del arg407_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf101 = extern_kernels.convolution(buf100, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf101, (8, 40, 14, 14), (7840, 1, 560, 40))
    del arg147_1
    buf102 = buf94; del buf94  # reuse
    cpp_fused_add_cat_47(c_void_p(buf102.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()))
    del arg148_1
    del arg149_1
    del arg409_1
    del arg410_1
    del buf100
    del buf101
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf103 = extern_kernels.convolution(buf102, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg150_1
    buf104 = buf103; del buf103  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf104.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()))
    del arg151_1
    del arg152_1
    del arg412_1
    del arg413_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf105 = extern_kernels.convolution(buf104, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf105, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg153_1
    buf106 = reinterpret_tensor(buf68, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf68  # reuse
    buf107 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf107, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf107  # reuse
    cpp_fused_cat_mean_49(c_void_p(buf108.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf106.data_ptr()))
    del arg154_1
    del arg155_1
    del arg415_1
    del arg416_1
    del buf104
    # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
    buf109 = extern_kernels.convolution(buf108, arg156_1, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf109, (8, 120, 1, 1), (120, 1, 120, 120))
    del arg156_1
    del arg157_1
    del buf108
    buf110 = buf109; del buf109  # reuse
    cpp_fused_relu_50(c_void_p(buf110.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.relu]
    buf111 = extern_kernels.convolution(buf110, arg158_1, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf111, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg158_1
    del arg159_1
    del buf110
    buf112 = buf106; del buf106  # reuse
    cpp_fused_hardsigmoid_mul_51(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()))
    del buf111
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___6_____3___se_gate, x_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf113 = extern_kernels.convolution(buf112, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (8, 56, 14, 14), (10976, 1, 784, 56))
    del arg160_1
    del buf112
    buf114 = buf113; del buf113  # reuse
    cpp_fused__native_batch_norm_legit_no_training_52(c_void_p(buf114.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()))
    del arg161_1
    del arg162_1
    del arg418_1
    del arg419_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf115 = extern_kernels.convolution(buf114, arg163_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
    assert_size_stride(buf115, (8, 56, 14, 14), (10976, 1, 784, 56))
    del arg163_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_0], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf102, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf116, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg166_1
    del buf102
    buf117 = buf116; del buf116  # reuse
    cpp_fused__native_batch_norm_legit_no_training_53(c_void_p(buf117.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()))
    del arg167_1
    del arg168_1
    del arg424_1
    del arg425_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1, getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf118 = extern_kernels.convolution(buf117, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg169_1
    del buf117
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_54(c_void_p(buf119.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()))
    del arg164_1
    del arg165_1
    del arg170_1
    del arg171_1
    del arg421_1
    del arg422_1
    del arg427_1
    del arg428_1
    del buf114
    del buf115
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg172_1
    buf121 = buf120; del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_55(c_void_p(buf121.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()))
    del arg173_1
    del arg174_1
    del arg430_1
    del arg431_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, arg175_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
    assert_size_stride(buf122, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg175_1
    buf123 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf125 = reinterpret_tensor(buf124, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf124  # reuse
    cpp_fused_cat_mean_56(c_void_p(buf125.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg176_1
    del arg177_1
    del arg433_1
    del arg434_1
    del buf121
    del buf122
    # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
    buf126 = extern_kernels.convolution(buf125, arg178_1, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf126, (8, 168, 1, 1), (168, 1, 168, 168))
    del arg178_1
    del arg179_1
    del buf125
    buf127 = buf126; del buf126  # reuse
    cpp_fused_relu_57(c_void_p(buf127.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.relu]
    buf128 = extern_kernels.convolution(buf127, arg180_1, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf128, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg180_1
    del arg181_1
    del buf127
    buf129 = buf123; del buf123  # reuse
    cpp_fused_hardsigmoid_mul_58(c_void_p(buf129.data_ptr()), c_void_p(buf128.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___6_____4___se_gate, x_43], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf130 = extern_kernels.convolution(buf129, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 56, 14, 14), (10976, 1, 784, 56))
    del arg182_1
    buf131 = buf130; del buf130  # reuse
    cpp_fused__native_batch_norm_legit_no_training_59(c_void_p(buf131.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()))
    del arg183_1
    del arg184_1
    del arg436_1
    del arg437_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, arg185_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
    assert_size_stride(buf132, (8, 56, 14, 14), (10976, 1, 784, 56))
    del arg185_1
    buf133 = buf119; del buf119  # reuse
    cpp_fused_add_cat_60(c_void_p(buf133.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()))
    del arg186_1
    del arg187_1
    del arg439_1
    del arg440_1
    del buf131
    del buf132
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg188_1
    buf135 = buf134; del buf134  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_61(c_void_p(buf135.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()))
    del arg189_1
    del arg190_1
    del arg442_1
    del arg443_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
    assert_size_stride(buf136, (8, 336, 14, 14), (65856, 1, 4704, 336))
    del arg191_1
    buf137 = buf129; del buf129  # reuse
    cpp_fused_cat_62(c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg192_1
    del arg193_1
    del arg445_1
    del arg446_1
    del buf135
    del buf136
    # Source Nodes: [cat_41, x_47], Original ATen: [aten.cat, aten.convolution]
    buf138 = extern_kernels.convolution(buf137, arg194_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf138, (8, 672, 7, 7), (32928, 1, 4704, 672))
    del arg194_1
    del buf137
    buf139 = buf138; del buf138  # reuse
    buf140 = reinterpret_tensor(buf128, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf128  # reuse
    buf141 = reinterpret_tensor(buf140, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_63(c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()))
    del arg195_1
    del arg196_1
    del arg448_1
    del arg449_1
    # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
    buf142 = extern_kernels.convolution(buf141, arg197_1, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf142, (8, 168, 1, 1), (168, 1, 168, 168))
    del arg197_1
    del arg198_1
    del buf141
    buf143 = buf142; del buf142  # reuse
    cpp_fused_relu_64(c_void_p(buf143.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.relu]
    buf144 = extern_kernels.convolution(buf143, arg199_1, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf144, (8, 672, 1, 1), (672, 1, 672, 672))
    del arg199_1
    del arg200_1
    del buf143
    buf145 = buf139; del buf139  # reuse
    cpp_fused_hardsigmoid_mul_65(c_void_p(buf145.data_ptr()), c_void_p(buf144.data_ptr()))
    del buf144
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___7_____0___se_gate, x_49], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf146 = extern_kernels.convolution(buf145, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg201_1
    del buf145
    buf147 = buf146; del buf146  # reuse
    cpp_fused__native_batch_norm_legit_no_training_66(c_void_p(buf147.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()))
    del arg202_1
    del arg203_1
    del arg451_1
    del arg452_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf147, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf148, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg204_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
    buf149 = extern_kernels.convolution(buf133, arg207_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf149, (8, 112, 7, 7), (5488, 1, 784, 112))
    del arg207_1
    del buf133
    buf150 = buf149; del buf149  # reuse
    cpp_fused__native_batch_norm_legit_no_training_67(c_void_p(buf150.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()))
    del arg208_1
    del arg209_1
    del arg457_1
    del arg458_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1, getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
    buf151 = extern_kernels.convolution(buf150, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 160, 7, 7), (7840, 1, 1120, 160))
    del arg210_1
    del buf150
    buf152 = buf151; del buf151  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_68(c_void_p(buf152.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg460_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()))
    del arg205_1
    del arg206_1
    del arg211_1
    del arg212_1
    del arg454_1
    del arg455_1
    del arg460_1
    del arg461_1
    del buf147
    del buf148
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf153 = extern_kernels.convolution(buf152, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf153, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg213_1
    buf154 = buf153; del buf153  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_69(c_void_p(buf154.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg463_1
    del arg464_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf155, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg216_1
    buf156 = reinterpret_tensor(buf105, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf105  # reuse
    cpp_fused_cat_70(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg217_1
    del arg218_1
    del arg466_1
    del arg467_1
    del buf154
    del buf155
    # Source Nodes: [cat_39, getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf157 = extern_kernels.convolution(buf156, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf157, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg219_1
    buf158 = buf157; del buf157  # reuse
    cpp_fused__native_batch_norm_legit_no_training_71(c_void_p(buf158.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()))
    del arg220_1
    del arg221_1
    del arg469_1
    del arg470_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf159 = extern_kernels.convolution(buf158, arg222_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf159, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg222_1
    buf160 = buf152; del buf152  # reuse
    cpp_fused_add_cat_72(c_void_p(buf160.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()))
    del arg223_1
    del arg224_1
    del arg472_1
    del arg473_1
    del buf158
    del buf159
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf161 = extern_kernels.convolution(buf160, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf161, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg225_1
    buf162 = buf161; del buf161  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_73(c_void_p(buf162.data_ptr()), c_void_p(arg475_1.data_ptr()), c_void_p(arg476_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg475_1
    del arg476_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf163 = extern_kernels.convolution(buf162, arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf163, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg228_1
    buf164 = buf156; del buf156  # reuse
    buf165 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf166 = reinterpret_tensor(buf165, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf165  # reuse
    cpp_fused_cat_mean_74(c_void_p(buf166.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg478_1.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf164.data_ptr()))
    del arg229_1
    del arg230_1
    del arg478_1
    del arg479_1
    del buf162
    del buf163
    # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
    buf167 = extern_kernels.convolution(buf166, arg231_1, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf167, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg231_1
    del arg232_1
    del buf166
    buf168 = buf167; del buf167  # reuse
    cpp_fused_relu_75(c_void_p(buf168.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.relu]
    buf169 = extern_kernels.convolution(buf168, arg233_1, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf169, (8, 960, 1, 1), (960, 1, 960, 960))
    del arg233_1
    del arg234_1
    del buf168
    buf170 = buf164; del buf164  # reuse
    cpp_fused_hardsigmoid_mul_76(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___8_____1___se_gate, x_56], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf171 = extern_kernels.convolution(buf170, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf171, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg235_1
    buf172 = buf171; del buf171  # reuse
    cpp_fused__native_batch_norm_legit_no_training_77(c_void_p(buf172.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(arg482_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()))
    del arg236_1
    del arg237_1
    del arg481_1
    del arg482_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf173 = extern_kernels.convolution(buf172, arg238_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf173, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg238_1
    buf174 = buf160; del buf160  # reuse
    cpp_fused_add_cat_78(c_void_p(buf174.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg484_1.data_ptr()), c_void_p(arg485_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg484_1
    del arg485_1
    del buf172
    del buf173
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf174, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg241_1
    buf176 = buf175; del buf175  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_79(c_void_p(buf176.data_ptr()), c_void_p(arg487_1.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg487_1
    del arg488_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf177 = extern_kernels.convolution(buf176, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf177, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg244_1
    buf178 = buf170; del buf170  # reuse
    cpp_fused_cat_80(c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(arg491_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg245_1
    del arg246_1
    del arg490_1
    del arg491_1
    del buf176
    del buf177
    # Source Nodes: [cat_35, getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0], Original ATen: [aten.cat, aten.convolution]
    buf179 = extern_kernels.convolution(buf178, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf179, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg247_1
    buf180 = buf179; del buf179  # reuse
    cpp_fused__native_batch_norm_legit_no_training_81(c_void_p(buf180.data_ptr()), c_void_p(arg493_1.data_ptr()), c_void_p(arg494_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()))
    del arg248_1
    del arg249_1
    del arg493_1
    del arg494_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf181 = extern_kernels.convolution(buf180, arg250_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf181, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg250_1
    buf182 = buf174; del buf174  # reuse
    cpp_fused_add_cat_82(c_void_p(buf182.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg496_1.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg496_1
    del arg497_1
    del buf180
    del buf181
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf183 = extern_kernels.convolution(buf182, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf183, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg253_1
    buf184 = buf183; del buf183  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_83(c_void_p(buf184.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()))
    del arg254_1
    del arg255_1
    del arg499_1
    del arg500_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf185 = extern_kernels.convolution(buf184, arg256_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf185, (8, 480, 7, 7), (23520, 1, 3360, 480))
    del arg256_1
    buf186 = buf178; del buf178  # reuse
    buf187 = reinterpret_tensor(buf169, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf169  # reuse
    buf188 = reinterpret_tensor(buf187, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf187  # reuse
    cpp_fused_cat_mean_84(c_void_p(buf188.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg257_1
    del arg258_1
    del arg502_1
    del arg503_1
    del buf184
    del buf185
    # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
    buf189 = extern_kernels.convolution(buf188, arg259_1, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf189, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg259_1
    del arg260_1
    del buf188
    buf190 = buf189; del buf189  # reuse
    cpp_fused_relu_85(c_void_p(buf190.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.relu]
    buf191 = extern_kernels.convolution(buf190, arg261_1, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf191, (8, 960, 1, 1), (960, 1, 960, 960))
    del arg261_1
    del arg262_1
    del buf190
    buf192 = buf186; del buf186  # reuse
    cpp_fused_hardsigmoid_mul_86(c_void_p(buf192.data_ptr()), c_void_p(buf191.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0, getattr_getattr_l__mod___blocks___8_____3___se_gate, x_63], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mul]
    buf193 = extern_kernels.convolution(buf192, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf193, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg263_1
    del buf192
    buf194 = buf193; del buf193  # reuse
    cpp_fused__native_batch_norm_legit_no_training_87(c_void_p(buf194.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()))
    del arg264_1
    del arg265_1
    del arg505_1
    del arg506_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf195 = extern_kernels.convolution(buf194, arg266_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf195, (8, 80, 7, 7), (3920, 1, 560, 80))
    del arg266_1
    buf196 = buf182; del buf182  # reuse
    cpp_fused_add_cat_88(c_void_p(buf196.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg508_1.data_ptr()), c_void_p(arg509_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()))
    del arg267_1
    del arg268_1
    del arg508_1
    del arg509_1
    del buf194
    del buf195
    # Source Nodes: [cat_32, shortcut_16, x_66], Original ATen: [aten.add, aten.cat, aten.convolution]
    buf197 = extern_kernels.convolution(buf196, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf197, (8, 960, 7, 7), (47040, 1, 6720, 960))
    del arg269_1
    del buf196
    buf198 = reinterpret_tensor(buf191, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf191  # reuse
    buf199 = reinterpret_tensor(buf198, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf198  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_89(c_void_p(buf199.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg272_1
    del arg273_1
    del buf197
    # Source Nodes: [x_67, x_72, x_73, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
    buf200 = extern_kernels.convolution(buf199, arg270_1, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf200, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    del arg270_1
    del arg271_1
    del buf199
    buf201 = reinterpret_tensor(buf200, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf200  # reuse
    cpp_fused_relu_90(c_void_p(buf201.data_ptr()))
    buf202 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg3_1, reinterpret_tensor(buf201, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg2_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf202)
    del arg2_1
    del arg3_1
    return (buf202, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg277_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg280_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg283_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg286_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg289_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg292_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg295_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg298_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg301_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg304_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg307_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg310_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg313_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg316_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg319_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg322_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg325_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg328_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg331_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg334_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg337_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg340_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg343_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg346_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg349_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg352_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg355_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg358_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg361_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg364_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg367_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg370_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg373_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg376_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg379_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg382_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg385_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg388_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg391_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg394_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg397_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg400_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg403_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg406_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg409_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg412_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg415_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg418_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg421_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg424_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg427_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg430_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg433_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg436_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg439_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg442_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg445_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg448_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg451_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg454_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg457_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg460_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg463_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg466_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg469_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg472_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg475_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg478_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg481_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg484_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg487_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg490_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg493_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg496_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg499_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg502_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg505_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg508_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg511_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
