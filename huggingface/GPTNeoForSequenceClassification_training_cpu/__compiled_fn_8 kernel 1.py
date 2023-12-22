
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


cpp_fused_new_zeros_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(2048.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_11 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_19 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_59 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_67 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_75 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_91 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_99 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_107 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_115 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_123 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_129 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_131 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_137 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_139 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_145 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_147 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_153 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_155 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_161 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_162 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_163 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_169 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_171 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_172 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_175 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_177 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_179 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_182 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_185 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_186 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_187 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_188 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_scalar_tensor_where_189 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_191 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_192 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = in_ptr6[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    auto tmp24 = static_cast<int>(-1);
                    auto tmp25 = tmp23 == tmp24;
                    auto tmp26 = static_cast<float>(0.0);
                    auto tmp27 = to_float_mask(tmp25);
                    auto tmp28 = at::vec::Vectorized<float>(tmp26);
                    auto tmp29 = decltype(tmp28)::blendv(tmp22, tmp28, tmp27);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    tmp29.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<bool>(0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 ? tmp2 : tmp0;
                in_out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_194 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(102926336L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, view, view_1, mul, view_2, slice_4, view_18, mul_2, view_20, addmm_1, tanh, view_22, mul_8, view_24, slice_8, view_40, mul_10, view_42, addmm_4, tanh_1, view_44, mul_16, view_46, slice_12, view_62, mul_18, view_64, addmm_7, tanh_2, view_66, mul_24, view_68, slice_16, view_84, mul_26, view_86, addmm_10, tanh_3, view_88, mul_32, view_90, slice_20, view_106, mul_34, view_108, addmm_13, tanh_4, view_110, mul_40, view_112, slice_24, view_128, mul_42, view_130, addmm_16, tanh_5, view_132, mul_48, view_134, slice_28, view_150, mul_50, view_152, addmm_19, tanh_6, view_154, mul_56, view_156, slice_32, view_172, mul_58, view_174, addmm_22, tanh_7, view_176, mul_64, view_178, slice_36, view_194, mul_66, view_196, addmm_25, tanh_8, view_198, mul_72, view_200, slice_40, view_216, mul_74, view_218, addmm_28, tanh_9, view_220, mul_80, view_222, slice_44, view_238, mul_82, view_240, addmm_31, tanh_10, view_242, mul_88, view_244, slice_48, view_260, mul_90, view_262, addmm_34, tanh_11, view_264, mul_96, view_266, slice_52, view_282, mul_98, view_284, addmm_37, tanh_12, view_286, mul_104, view_288, slice_56, view_304, mul_106, view_306, addmm_40, tanh_13, view_308, mul_112, view_310, slice_60, view_326, mul_114, view_328, addmm_43, tanh_14, view_330, mul_120, view_332, slice_64, view_348, mul_122, view_350, addmm_46, tanh_15, view_352, mul_128, view_354, slice_68, view_370, mul_130, view_372, addmm_49, tanh_16, view_374, mul_136, view_376, slice_72, view_392, mul_138, view_394, addmm_52, tanh_17, view_396, mul_144, view_398, slice_76, view_414, mul_146, view_416, addmm_55, tanh_18, view_418, mul_152, view_420, slice_80, view_436, mul_154, view_438, addmm_58, tanh_19, view_440, mul_160, view_442, slice_84, view_458, mul_162, view_460, addmm_61, tanh_20, view_462, mul_168, view_464, slice_88, view_480, mul_170, view_482, addmm_64, tanh_21, view_484, mul_176, view_486, slice_92, view_502, mul_178, view_504, addmm_67, tanh_22, view_506, mul_184, view_508, slice_96, view_524, mul_186, view_526, addmm_70, tanh_23, view_528, mul_192, view_531, sub_73, full_default_24, permute_267, div_24, permute_269, permute_273, div_25, permute_277, permute_282, permute_283, alias_49, permute_284, permute_285, permute_292, permute_296, permute_300, div_26, permute_302, permute_306, div_27, permute_310, permute_315, permute_316, alias_51, permute_317, permute_318, permute_325, permute_329, permute_333, div_28, permute_335, permute_339, div_29, permute_343, permute_348, permute_349, alias_53, permute_350, permute_351, permute_358, permute_362, permute_366, div_30, permute_368, permute_372, div_31, permute_376, permute_381, permute_382, alias_55, permute_383, permute_384, permute_391, permute_395, permute_399, div_32, permute_401, permute_405, div_33, permute_409, permute_414, permute_415, alias_57, permute_416, permute_417, permute_424, permute_428, permute_432, div_34, permute_434, permute_438, div_35, permute_442, permute_447, permute_448, alias_59, permute_449, permute_450, permute_457, permute_461, permute_465, div_36, permute_467, permute_471, div_37, permute_475, permute_480, permute_481, alias_61, permute_482, permute_483, permute_490, permute_494, permute_498, div_38, permute_500, permute_504, div_39, permute_508, permute_513, permute_514, alias_63, permute_515, permute_516, permute_523, permute_527, permute_531, div_40, permute_533, permute_537, div_41, permute_541, permute_546, permute_547, alias_65, permute_548, permute_549, permute_556, permute_560, permute_564, div_42, permute_566, permute_570, div_43, permute_574, permute_579, permute_580, alias_67, permute_581, permute_582, permute_589, permute_593, permute_597, div_44, permute_599, permute_603, div_45, permute_607, permute_612, permute_613, alias_69, permute_614, permute_615, permute_622, permute_626, permute_630, div_46, permute_632, permute_636, div_47, permute_640, permute_645, permute_646, alias_71, permute_647, permute_648, permute_655, permute_659, permute_663, div_48, permute_665, permute_669, div_49, permute_673, permute_678, permute_679, alias_73, permute_680, permute_681, permute_688, permute_692, permute_696, div_50, permute_698, permute_702, div_51, permute_706, permute_711, permute_712, alias_75, permute_713, permute_714, permute_721, permute_725, permute_729, div_52, permute_731, permute_735, div_53, permute_739, permute_744, permute_745, alias_77, permute_746, permute_747, permute_754, permute_758, permute_762, div_54, permute_764, permute_768, div_55, permute_772, permute_777, permute_778, alias_79, permute_779, permute_780, permute_787, permute_791, permute_795, div_56, permute_797, permute_801, div_57, permute_805, permute_810, permute_811, alias_81, permute_812, permute_813, permute_820, permute_824, permute_828, div_58, permute_830, permute_834, div_59, permute_838, permute_843, permute_844, alias_83, permute_845, permute_846, permute_853, permute_857, permute_861, div_60, permute_863, permute_867, div_61, permute_871, permute_876, permute_877, alias_85, permute_878, permute_879, permute_886, permute_890, permute_894, div_62, permute_896, permute_900, div_63, permute_904, permute_909, permute_910, alias_87, permute_911, permute_912, permute_919, permute_923, permute_927, div_64, permute_929, permute_933, div_65, permute_937, permute_942, permute_943, alias_89, permute_944, permute_945, permute_952, permute_956, permute_960, div_66, permute_962, permute_966, div_67, permute_970, permute_975, permute_976, alias_91, permute_977, permute_978, permute_985, permute_989, permute_993, div_68, permute_995, permute_999, div_69, permute_1003, permute_1008, permute_1009, alias_93, permute_1010, permute_1011, permute_1018, permute_1022, permute_1026, div_70, permute_1028, permute_1032, div_71, permute_1036, permute_1041, permute_1042, alias_95, permute_1043, permute_1044, permute_1051, permute_1055, permute_1059, div_72, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50 = args
    args.clear()
    assert_size_stride(primals_3, (2048, ), (1, ))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_16, (2048, ), (1, ))
    assert_size_stride(primals_23, (2048, ), (1, ))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_42, (2048, ), (1, ))
    assert_size_stride(primals_49, (2048, ), (1, ))
    assert_size_stride(primals_55, (2048, ), (1, ))
    assert_size_stride(primals_62, (2048, ), (1, ))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_75, (2048, ), (1, ))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, ), (1, ))
    assert_size_stride(primals_94, (2048, ), (1, ))
    assert_size_stride(primals_101, (2048, ), (1, ))
    assert_size_stride(primals_107, (2048, ), (1, ))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_120, (2048, ), (1, ))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_133, (2048, ), (1, ))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_146, (2048, ), (1, ))
    assert_size_stride(primals_153, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_166, (2048, ), (1, ))
    assert_size_stride(primals_172, (2048, ), (1, ))
    assert_size_stride(primals_179, (2048, ), (1, ))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_192, (2048, ), (1, ))
    assert_size_stride(primals_198, (2048, ), (1, ))
    assert_size_stride(primals_205, (2048, ), (1, ))
    assert_size_stride(primals_211, (2048, ), (1, ))
    assert_size_stride(primals_218, (2048, ), (1, ))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_231, (2048, ), (1, ))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_244, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, ), (1, ))
    assert_size_stride(primals_257, (2048, ), (1, ))
    assert_size_stride(primals_263, (2048, ), (1, ))
    assert_size_stride(primals_270, (2048, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_283, (2048, ), (1, ))
    assert_size_stride(primals_289, (2048, ), (1, ))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_315, (2048, ), (1, ))
    assert_size_stride(view, (1, 128), (128, 1))
    assert_size_stride(view_1, (1, 128), (128, 1))
    assert_size_stride(mul, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_2, (128, 2048), (2048, 1))
    assert_size_stride(slice_4, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_18, (128, 2048), (2048, 1))
    assert_size_stride(mul_2, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_20, (128, 2048), (2048, 1))
    assert_size_stride(addmm_1, (128, 8192), (8192, 1))
    assert_size_stride(tanh, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_22, (128, 8192), (8192, 1))
    assert_size_stride(mul_8, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_24, (128, 2048), (2048, 1))
    assert_size_stride(slice_8, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_40, (128, 2048), (2048, 1))
    assert_size_stride(mul_10, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_42, (128, 2048), (2048, 1))
    assert_size_stride(addmm_4, (128, 8192), (8192, 1))
    assert_size_stride(tanh_1, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_44, (128, 8192), (8192, 1))
    assert_size_stride(mul_16, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_46, (128, 2048), (2048, 1))
    assert_size_stride(slice_12, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_62, (128, 2048), (2048, 1))
    assert_size_stride(mul_18, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_64, (128, 2048), (2048, 1))
    assert_size_stride(addmm_7, (128, 8192), (8192, 1))
    assert_size_stride(tanh_2, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_66, (128, 8192), (8192, 1))
    assert_size_stride(mul_24, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_68, (128, 2048), (2048, 1))
    assert_size_stride(slice_16, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_84, (128, 2048), (2048, 1))
    assert_size_stride(mul_26, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_86, (128, 2048), (2048, 1))
    assert_size_stride(addmm_10, (128, 8192), (8192, 1))
    assert_size_stride(tanh_3, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_88, (128, 8192), (8192, 1))
    assert_size_stride(mul_32, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_90, (128, 2048), (2048, 1))
    assert_size_stride(slice_20, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_106, (128, 2048), (2048, 1))
    assert_size_stride(mul_34, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_108, (128, 2048), (2048, 1))
    assert_size_stride(addmm_13, (128, 8192), (8192, 1))
    assert_size_stride(tanh_4, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_110, (128, 8192), (8192, 1))
    assert_size_stride(mul_40, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_112, (128, 2048), (2048, 1))
    assert_size_stride(slice_24, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_128, (128, 2048), (2048, 1))
    assert_size_stride(mul_42, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_130, (128, 2048), (2048, 1))
    assert_size_stride(addmm_16, (128, 8192), (8192, 1))
    assert_size_stride(tanh_5, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_132, (128, 8192), (8192, 1))
    assert_size_stride(mul_48, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_134, (128, 2048), (2048, 1))
    assert_size_stride(slice_28, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_150, (128, 2048), (2048, 1))
    assert_size_stride(mul_50, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_152, (128, 2048), (2048, 1))
    assert_size_stride(addmm_19, (128, 8192), (8192, 1))
    assert_size_stride(tanh_6, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_154, (128, 8192), (8192, 1))
    assert_size_stride(mul_56, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_156, (128, 2048), (2048, 1))
    assert_size_stride(slice_32, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_172, (128, 2048), (2048, 1))
    assert_size_stride(mul_58, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_174, (128, 2048), (2048, 1))
    assert_size_stride(addmm_22, (128, 8192), (8192, 1))
    assert_size_stride(tanh_7, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_176, (128, 8192), (8192, 1))
    assert_size_stride(mul_64, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_178, (128, 2048), (2048, 1))
    assert_size_stride(slice_36, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_194, (128, 2048), (2048, 1))
    assert_size_stride(mul_66, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_196, (128, 2048), (2048, 1))
    assert_size_stride(addmm_25, (128, 8192), (8192, 1))
    assert_size_stride(tanh_8, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_198, (128, 8192), (8192, 1))
    assert_size_stride(mul_72, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_200, (128, 2048), (2048, 1))
    assert_size_stride(slice_40, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_216, (128, 2048), (2048, 1))
    assert_size_stride(mul_74, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_218, (128, 2048), (2048, 1))
    assert_size_stride(addmm_28, (128, 8192), (8192, 1))
    assert_size_stride(tanh_9, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_220, (128, 8192), (8192, 1))
    assert_size_stride(mul_80, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_222, (128, 2048), (2048, 1))
    assert_size_stride(slice_44, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_238, (128, 2048), (2048, 1))
    assert_size_stride(mul_82, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_240, (128, 2048), (2048, 1))
    assert_size_stride(addmm_31, (128, 8192), (8192, 1))
    assert_size_stride(tanh_10, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_242, (128, 8192), (8192, 1))
    assert_size_stride(mul_88, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_244, (128, 2048), (2048, 1))
    assert_size_stride(slice_48, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_260, (128, 2048), (2048, 1))
    assert_size_stride(mul_90, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_262, (128, 2048), (2048, 1))
    assert_size_stride(addmm_34, (128, 8192), (8192, 1))
    assert_size_stride(tanh_11, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_264, (128, 8192), (8192, 1))
    assert_size_stride(mul_96, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_266, (128, 2048), (2048, 1))
    assert_size_stride(slice_52, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_282, (128, 2048), (2048, 1))
    assert_size_stride(mul_98, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_284, (128, 2048), (2048, 1))
    assert_size_stride(addmm_37, (128, 8192), (8192, 1))
    assert_size_stride(tanh_12, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_286, (128, 8192), (8192, 1))
    assert_size_stride(mul_104, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_288, (128, 2048), (2048, 1))
    assert_size_stride(slice_56, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_304, (128, 2048), (2048, 1))
    assert_size_stride(mul_106, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_306, (128, 2048), (2048, 1))
    assert_size_stride(addmm_40, (128, 8192), (8192, 1))
    assert_size_stride(tanh_13, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_308, (128, 8192), (8192, 1))
    assert_size_stride(mul_112, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_310, (128, 2048), (2048, 1))
    assert_size_stride(slice_60, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_326, (128, 2048), (2048, 1))
    assert_size_stride(mul_114, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_328, (128, 2048), (2048, 1))
    assert_size_stride(addmm_43, (128, 8192), (8192, 1))
    assert_size_stride(tanh_14, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_330, (128, 8192), (8192, 1))
    assert_size_stride(mul_120, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_332, (128, 2048), (2048, 1))
    assert_size_stride(slice_64, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_348, (128, 2048), (2048, 1))
    assert_size_stride(mul_122, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_350, (128, 2048), (2048, 1))
    assert_size_stride(addmm_46, (128, 8192), (8192, 1))
    assert_size_stride(tanh_15, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_352, (128, 8192), (8192, 1))
    assert_size_stride(mul_128, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_354, (128, 2048), (2048, 1))
    assert_size_stride(slice_68, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_370, (128, 2048), (2048, 1))
    assert_size_stride(mul_130, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_372, (128, 2048), (2048, 1))
    assert_size_stride(addmm_49, (128, 8192), (8192, 1))
    assert_size_stride(tanh_16, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_374, (128, 8192), (8192, 1))
    assert_size_stride(mul_136, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_376, (128, 2048), (2048, 1))
    assert_size_stride(slice_72, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_392, (128, 2048), (2048, 1))
    assert_size_stride(mul_138, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_394, (128, 2048), (2048, 1))
    assert_size_stride(addmm_52, (128, 8192), (8192, 1))
    assert_size_stride(tanh_17, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_396, (128, 8192), (8192, 1))
    assert_size_stride(mul_144, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_398, (128, 2048), (2048, 1))
    assert_size_stride(slice_76, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_414, (128, 2048), (2048, 1))
    assert_size_stride(mul_146, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_416, (128, 2048), (2048, 1))
    assert_size_stride(addmm_55, (128, 8192), (8192, 1))
    assert_size_stride(tanh_18, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_418, (128, 8192), (8192, 1))
    assert_size_stride(mul_152, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_420, (128, 2048), (2048, 1))
    assert_size_stride(slice_80, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_436, (128, 2048), (2048, 1))
    assert_size_stride(mul_154, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_438, (128, 2048), (2048, 1))
    assert_size_stride(addmm_58, (128, 8192), (8192, 1))
    assert_size_stride(tanh_19, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_440, (128, 8192), (8192, 1))
    assert_size_stride(mul_160, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_442, (128, 2048), (2048, 1))
    assert_size_stride(slice_84, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_458, (128, 2048), (2048, 1))
    assert_size_stride(mul_162, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_460, (128, 2048), (2048, 1))
    assert_size_stride(addmm_61, (128, 8192), (8192, 1))
    assert_size_stride(tanh_20, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_462, (128, 8192), (8192, 1))
    assert_size_stride(mul_168, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_464, (128, 2048), (2048, 1))
    assert_size_stride(slice_88, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_480, (128, 2048), (2048, 1))
    assert_size_stride(mul_170, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_482, (128, 2048), (2048, 1))
    assert_size_stride(addmm_64, (128, 8192), (8192, 1))
    assert_size_stride(tanh_21, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_484, (128, 8192), (8192, 1))
    assert_size_stride(mul_176, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_486, (128, 2048), (2048, 1))
    assert_size_stride(slice_92, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_502, (128, 2048), (2048, 1))
    assert_size_stride(mul_178, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_504, (128, 2048), (2048, 1))
    assert_size_stride(addmm_67, (128, 8192), (8192, 1))
    assert_size_stride(tanh_22, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_506, (128, 8192), (8192, 1))
    assert_size_stride(mul_184, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_508, (128, 2048), (2048, 1))
    assert_size_stride(slice_96, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_524, (128, 2048), (2048, 1))
    assert_size_stride(mul_186, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_526, (128, 2048), (2048, 1))
    assert_size_stride(addmm_70, (128, 8192), (8192, 1))
    assert_size_stride(tanh_23, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_528, (128, 8192), (8192, 1))
    assert_size_stride(mul_192, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_531, (128, 2048), (2048, 1))
    assert_size_stride(sub_73, (1, ), (1, ))
    assert_size_stride(full_default_24, (1, ), (1, ))
    assert_size_stride(permute_267, (2, 2048), (2048, 1))
    assert_size_stride(div_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_269, (2048, 8192), (8192, 1))
    assert_size_stride(permute_273, (8192, 2048), (2048, 1))
    assert_size_stride(div_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_277, (2048, 2048), (2048, 1))
    assert_size_stride(permute_282, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_283, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_49, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_284, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_285, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_292, (2048, 2048), (2048, 1))
    assert_size_stride(permute_296, (2048, 2048), (2048, 1))
    assert_size_stride(permute_300, (2048, 2048), (2048, 1))
    assert_size_stride(div_26, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_302, (2048, 8192), (8192, 1))
    assert_size_stride(permute_306, (8192, 2048), (2048, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_310, (2048, 2048), (2048, 1))
    assert_size_stride(permute_315, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_316, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_51, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_317, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_318, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_325, (2048, 2048), (2048, 1))
    assert_size_stride(permute_329, (2048, 2048), (2048, 1))
    assert_size_stride(permute_333, (2048, 2048), (2048, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_335, (2048, 8192), (8192, 1))
    assert_size_stride(permute_339, (8192, 2048), (2048, 1))
    assert_size_stride(div_29, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_343, (2048, 2048), (2048, 1))
    assert_size_stride(permute_348, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_349, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_53, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_350, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_351, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_358, (2048, 2048), (2048, 1))
    assert_size_stride(permute_362, (2048, 2048), (2048, 1))
    assert_size_stride(permute_366, (2048, 2048), (2048, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_368, (2048, 8192), (8192, 1))
    assert_size_stride(permute_372, (8192, 2048), (2048, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_376, (2048, 2048), (2048, 1))
    assert_size_stride(permute_381, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_382, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_55, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_383, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_384, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_391, (2048, 2048), (2048, 1))
    assert_size_stride(permute_395, (2048, 2048), (2048, 1))
    assert_size_stride(permute_399, (2048, 2048), (2048, 1))
    assert_size_stride(div_32, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_401, (2048, 8192), (8192, 1))
    assert_size_stride(permute_405, (8192, 2048), (2048, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_409, (2048, 2048), (2048, 1))
    assert_size_stride(permute_414, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_415, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_57, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_416, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_417, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_424, (2048, 2048), (2048, 1))
    assert_size_stride(permute_428, (2048, 2048), (2048, 1))
    assert_size_stride(permute_432, (2048, 2048), (2048, 1))
    assert_size_stride(div_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_434, (2048, 8192), (8192, 1))
    assert_size_stride(permute_438, (8192, 2048), (2048, 1))
    assert_size_stride(div_35, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_442, (2048, 2048), (2048, 1))
    assert_size_stride(permute_447, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_448, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_59, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_449, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_450, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_457, (2048, 2048), (2048, 1))
    assert_size_stride(permute_461, (2048, 2048), (2048, 1))
    assert_size_stride(permute_465, (2048, 2048), (2048, 1))
    assert_size_stride(div_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_467, (2048, 8192), (8192, 1))
    assert_size_stride(permute_471, (8192, 2048), (2048, 1))
    assert_size_stride(div_37, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_475, (2048, 2048), (2048, 1))
    assert_size_stride(permute_480, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_481, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_61, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_482, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_483, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_490, (2048, 2048), (2048, 1))
    assert_size_stride(permute_494, (2048, 2048), (2048, 1))
    assert_size_stride(permute_498, (2048, 2048), (2048, 1))
    assert_size_stride(div_38, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_500, (2048, 8192), (8192, 1))
    assert_size_stride(permute_504, (8192, 2048), (2048, 1))
    assert_size_stride(div_39, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_508, (2048, 2048), (2048, 1))
    assert_size_stride(permute_513, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_514, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_63, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_515, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_516, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_523, (2048, 2048), (2048, 1))
    assert_size_stride(permute_527, (2048, 2048), (2048, 1))
    assert_size_stride(permute_531, (2048, 2048), (2048, 1))
    assert_size_stride(div_40, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_533, (2048, 8192), (8192, 1))
    assert_size_stride(permute_537, (8192, 2048), (2048, 1))
    assert_size_stride(div_41, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_541, (2048, 2048), (2048, 1))
    assert_size_stride(permute_546, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_547, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_65, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_548, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_549, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_556, (2048, 2048), (2048, 1))
    assert_size_stride(permute_560, (2048, 2048), (2048, 1))
    assert_size_stride(permute_564, (2048, 2048), (2048, 1))
    assert_size_stride(div_42, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_566, (2048, 8192), (8192, 1))
    assert_size_stride(permute_570, (8192, 2048), (2048, 1))
    assert_size_stride(div_43, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_574, (2048, 2048), (2048, 1))
    assert_size_stride(permute_579, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_580, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_67, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_581, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_582, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_589, (2048, 2048), (2048, 1))
    assert_size_stride(permute_593, (2048, 2048), (2048, 1))
    assert_size_stride(permute_597, (2048, 2048), (2048, 1))
    assert_size_stride(div_44, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_599, (2048, 8192), (8192, 1))
    assert_size_stride(permute_603, (8192, 2048), (2048, 1))
    assert_size_stride(div_45, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_607, (2048, 2048), (2048, 1))
    assert_size_stride(permute_612, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_613, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_69, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_614, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_615, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_622, (2048, 2048), (2048, 1))
    assert_size_stride(permute_626, (2048, 2048), (2048, 1))
    assert_size_stride(permute_630, (2048, 2048), (2048, 1))
    assert_size_stride(div_46, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_632, (2048, 8192), (8192, 1))
    assert_size_stride(permute_636, (8192, 2048), (2048, 1))
    assert_size_stride(div_47, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_640, (2048, 2048), (2048, 1))
    assert_size_stride(permute_645, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_646, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_71, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_647, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_648, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_655, (2048, 2048), (2048, 1))
    assert_size_stride(permute_659, (2048, 2048), (2048, 1))
    assert_size_stride(permute_663, (2048, 2048), (2048, 1))
    assert_size_stride(div_48, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_665, (2048, 8192), (8192, 1))
    assert_size_stride(permute_669, (8192, 2048), (2048, 1))
    assert_size_stride(div_49, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_673, (2048, 2048), (2048, 1))
    assert_size_stride(permute_678, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_679, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_73, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_680, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_681, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_688, (2048, 2048), (2048, 1))
    assert_size_stride(permute_692, (2048, 2048), (2048, 1))
    assert_size_stride(permute_696, (2048, 2048), (2048, 1))
    assert_size_stride(div_50, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_698, (2048, 8192), (8192, 1))
    assert_size_stride(permute_702, (8192, 2048), (2048, 1))
    assert_size_stride(div_51, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_706, (2048, 2048), (2048, 1))
    assert_size_stride(permute_711, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_712, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_75, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_713, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_714, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_721, (2048, 2048), (2048, 1))
    assert_size_stride(permute_725, (2048, 2048), (2048, 1))
    assert_size_stride(permute_729, (2048, 2048), (2048, 1))
    assert_size_stride(div_52, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_731, (2048, 8192), (8192, 1))
    assert_size_stride(permute_735, (8192, 2048), (2048, 1))
    assert_size_stride(div_53, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_739, (2048, 2048), (2048, 1))
    assert_size_stride(permute_744, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_745, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_77, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_746, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_747, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_754, (2048, 2048), (2048, 1))
    assert_size_stride(permute_758, (2048, 2048), (2048, 1))
    assert_size_stride(permute_762, (2048, 2048), (2048, 1))
    assert_size_stride(div_54, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_764, (2048, 8192), (8192, 1))
    assert_size_stride(permute_768, (8192, 2048), (2048, 1))
    assert_size_stride(div_55, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_772, (2048, 2048), (2048, 1))
    assert_size_stride(permute_777, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_778, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_79, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_779, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_780, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_787, (2048, 2048), (2048, 1))
    assert_size_stride(permute_791, (2048, 2048), (2048, 1))
    assert_size_stride(permute_795, (2048, 2048), (2048, 1))
    assert_size_stride(div_56, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_797, (2048, 8192), (8192, 1))
    assert_size_stride(permute_801, (8192, 2048), (2048, 1))
    assert_size_stride(div_57, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_805, (2048, 2048), (2048, 1))
    assert_size_stride(permute_810, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_811, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_81, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_812, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_813, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_820, (2048, 2048), (2048, 1))
    assert_size_stride(permute_824, (2048, 2048), (2048, 1))
    assert_size_stride(permute_828, (2048, 2048), (2048, 1))
    assert_size_stride(div_58, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_830, (2048, 8192), (8192, 1))
    assert_size_stride(permute_834, (8192, 2048), (2048, 1))
    assert_size_stride(div_59, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_838, (2048, 2048), (2048, 1))
    assert_size_stride(permute_843, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_844, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_83, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_845, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_846, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_853, (2048, 2048), (2048, 1))
    assert_size_stride(permute_857, (2048, 2048), (2048, 1))
    assert_size_stride(permute_861, (2048, 2048), (2048, 1))
    assert_size_stride(div_60, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_863, (2048, 8192), (8192, 1))
    assert_size_stride(permute_867, (8192, 2048), (2048, 1))
    assert_size_stride(div_61, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_871, (2048, 2048), (2048, 1))
    assert_size_stride(permute_876, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_877, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_85, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_878, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_879, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_886, (2048, 2048), (2048, 1))
    assert_size_stride(permute_890, (2048, 2048), (2048, 1))
    assert_size_stride(permute_894, (2048, 2048), (2048, 1))
    assert_size_stride(div_62, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_896, (2048, 8192), (8192, 1))
    assert_size_stride(permute_900, (8192, 2048), (2048, 1))
    assert_size_stride(div_63, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_904, (2048, 2048), (2048, 1))
    assert_size_stride(permute_909, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_910, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_87, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_911, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_912, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_919, (2048, 2048), (2048, 1))
    assert_size_stride(permute_923, (2048, 2048), (2048, 1))
    assert_size_stride(permute_927, (2048, 2048), (2048, 1))
    assert_size_stride(div_64, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_929, (2048, 8192), (8192, 1))
    assert_size_stride(permute_933, (8192, 2048), (2048, 1))
    assert_size_stride(div_65, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_937, (2048, 2048), (2048, 1))
    assert_size_stride(permute_942, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_943, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_89, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_944, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_945, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_952, (2048, 2048), (2048, 1))
    assert_size_stride(permute_956, (2048, 2048), (2048, 1))
    assert_size_stride(permute_960, (2048, 2048), (2048, 1))
    assert_size_stride(div_66, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_962, (2048, 8192), (8192, 1))
    assert_size_stride(permute_966, (8192, 2048), (2048, 1))
    assert_size_stride(div_67, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_970, (2048, 2048), (2048, 1))
    assert_size_stride(permute_975, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_976, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_91, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_977, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_978, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_985, (2048, 2048), (2048, 1))
    assert_size_stride(permute_989, (2048, 2048), (2048, 1))
    assert_size_stride(permute_993, (2048, 2048), (2048, 1))
    assert_size_stride(div_68, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_995, (2048, 8192), (8192, 1))
    assert_size_stride(permute_999, (8192, 2048), (2048, 1))
    assert_size_stride(div_69, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1003, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1008, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1009, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_93, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_1010, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_1011, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_1018, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1022, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1026, (2048, 2048), (2048, 1))
    assert_size_stride(div_70, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1028, (2048, 8192), (8192, 1))
    assert_size_stride(permute_1032, (8192, 2048), (2048, 1))
    assert_size_stride(div_71, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1036, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1041, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1042, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_95, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_1043, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_1044, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_1051, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1055, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1059, (2048, 2048), (2048, 1))
    assert_size_stride(div_72, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(tangents_2, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_3, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_4, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_5, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_6, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_7, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_8, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_9, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_10, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_11, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_12, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_13, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_14, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_15, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_16, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_17, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_18, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_19, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_20, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_21, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_22, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_23, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_24, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_25, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_26, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_27, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_28, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_29, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_30, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_31, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_32, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_33, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_34, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_35, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_36, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_37, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_38, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_39, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_40, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_41, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_42, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_43, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_44, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_45, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_46, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_47, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_48, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_49, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_50, (1, 2), (2, 1))
    buf0 = empty((1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused_new_zeros_0(c_void_p(buf0.data_ptr()))
    aten.index_put_(buf0, [full_default_24, sub_73], tangents_50, True)
    del full_default_24
    del sub_73
    del tangents_50
    buf3 = empty((2, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (2, 128), (1, 2), 0), view_531, out=buf3)
    del view_531
    buf4 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (128, 2), (2, 1), 0), permute_267, out=buf4)
    del buf0
    del permute_267
    buf5 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf8 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf9 = empty((2048, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_1(c_void_p(tangents_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del div_24
    del mul_192
    del primals_315
    del tangents_1
    buf10 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (128, 2048), (2048, 1), 0), permute_269, out=buf10)
    del permute_269
    buf11 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (2048, 128), (1, 2048), 0), view_528, out=buf11)
    del view_528
    buf12 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf10, (1, 128, 8192), (1048576, 8192, 1), 0); del buf10  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_2(c_void_p(buf13.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(tanh_23.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_70
    del tanh_23
    buf14 = buf4; del buf4  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (128, 8192), (8192, 1), 0), permute_273, out=buf14)
    del permute_273
    buf15 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (8192, 128), (1, 8192), 0), view_526, out=buf15)
    del view_526
    buf16 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf17 = buf6; del buf6  # reuse
    buf18 = buf5; del buf5  # reuse
    buf19 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf20 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf14, (1, 128, 2048), (262144, 2048, 1), 0); del buf14  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_3(c_void_p(buf21.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(mul_186.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del div_25
    del mul_186
    del primals_309
    buf22 = reinterpret_tensor(buf7, (128, 2048), (2048, 1), 0); del buf7  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (128, 2048), (2048, 1), 0), permute_277, out=buf22)
    del permute_277
    buf23 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (2048, 128), (1, 2048), 0), view_524, out=buf23)
    del view_524
    buf24 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_4(c_void_p(buf21.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_282, reinterpret_tensor(buf22, (16, 128, 128), (128, 2048, 1), 0), out=buf25)
    del permute_282
    buf26 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (16, 128, 128), (128, 2048, 1), 0), permute_283, out=buf26)
    del permute_283
    buf27 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf28 = reinterpret_tensor(buf26, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf26  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_5(c_void_p(buf28.data_ptr()), c_void_p(alias_49.data_ptr()), c_void_p(slice_96.data_ptr()), c_void_p(buf27.data_ptr()))
    del alias_49
    del slice_96
    buf29 = reinterpret_tensor(buf22, (16, 128, 128), (16384, 128, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_284, reinterpret_tensor(buf28, (16, 128, 128), (16384, 128, 1), 0), out=buf29)
    del permute_284
    buf30 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (16, 128, 128), (16384, 128, 1), 0), permute_285, out=buf30)
    del permute_285
    buf31 = reinterpret_tensor(buf28, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf28  # reuse
    cpp_fused_clone_6(c_void_p(tangents_49.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf31.data_ptr()))
    del tangents_49
    buf32 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (2048, 128), (1, 2048), 0), view_508, out=buf32)
    buf33 = reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0); del buf25  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (128, 2048), (2048, 1), 0), permute_292, out=buf33)
    del permute_292
    buf34 = buf31; del buf31  # reuse
    cpp_fused_clone_7(c_void_p(tangents_48.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf34.data_ptr()))
    del tangents_48
    buf35 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (2048, 128), (1, 2048), 0), view_508, out=buf35)
    buf36 = reinterpret_tensor(buf29, (128, 2048), (2048, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (128, 2048), (2048, 1), 0), permute_296, out=buf36)
    del permute_296
    buf37 = reinterpret_tensor(buf34, (128, 2048), (2048, 1), 0); del buf34  # reuse
    cpp_fused_view_8(c_void_p(buf30.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (2048, 128), (1, 2048), 0), view_508, out=buf38)
    del view_508
    buf39 = reinterpret_tensor(buf30, (128, 2048), (2048, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf37, permute_300, out=buf39)
    del permute_300
    buf40 = buf18; del buf18  # reuse
    buf41 = buf17; del buf17  # reuse
    buf42 = reinterpret_tensor(buf27, (2048, ), (1, ), 0); del buf27  # reuse
    buf43 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf44 = buf21; del buf21  # reuse
    cpp_fused_add_native_layer_norm_backward_9(c_void_p(buf44.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(mul_184.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del div_26
    del mul_184
    del primals_302
    buf45 = reinterpret_tensor(buf13, (128, 8192), (8192, 1), 0); del buf13  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (128, 2048), (2048, 1), 0), permute_302, out=buf45)
    del permute_302
    buf46 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (2048, 128), (1, 2048), 0), view_506, out=buf46)
    del view_506
    buf47 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf48 = reinterpret_tensor(buf45, (1, 128, 8192), (1048576, 8192, 1), 0); del buf45  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_10(c_void_p(buf48.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(addmm_67.data_ptr()), c_void_p(tanh_22.data_ptr()), c_void_p(buf47.data_ptr()))
    del addmm_67
    del tanh_22
    buf49 = buf39; del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (128, 8192), (8192, 1), 0), permute_306, out=buf49)
    del permute_306
    buf50 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (8192, 128), (1, 8192), 0), view_504, out=buf50)
    del view_504
    buf51 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf52 = buf41; del buf41  # reuse
    buf53 = buf40; del buf40  # reuse
    buf54 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf55 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf56 = buf44; del buf44  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_11(c_void_p(buf56.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(mul_178.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del div_27
    del mul_178
    del primals_296
    buf57 = buf49; del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (128, 2048), (2048, 1), 0), permute_310, out=buf57)
    del permute_310
    buf58 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (2048, 128), (1, 2048), 0), view_502, out=buf58)
    del view_502
    buf59 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf56.data_ptr()), c_void_p(buf59.data_ptr()))
    buf60 = reinterpret_tensor(buf36, (16, 128, 128), (16384, 128, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_315, reinterpret_tensor(buf57, (16, 128, 128), (128, 2048, 1), 0), out=buf60)
    del permute_315
    buf61 = reinterpret_tensor(buf33, (16, 128, 128), (16384, 128, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf57, (16, 128, 128), (128, 2048, 1), 0), permute_316, out=buf61)
    del permute_316
    buf62 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf63 = reinterpret_tensor(buf61, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf61  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_13(c_void_p(buf63.data_ptr()), c_void_p(alias_51.data_ptr()), c_void_p(slice_92.data_ptr()), c_void_p(buf62.data_ptr()))
    del alias_51
    del slice_92
    buf64 = reinterpret_tensor(buf57, (16, 128, 128), (16384, 128, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_317, reinterpret_tensor(buf63, (16, 128, 128), (16384, 128, 1), 0), out=buf64)
    del permute_317
    buf65 = reinterpret_tensor(buf37, (16, 128, 128), (16384, 128, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (16, 128, 128), (16384, 128, 1), 0), permute_318, out=buf65)
    del permute_318
    buf66 = reinterpret_tensor(buf63, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf63  # reuse
    cpp_fused_clone_14(c_void_p(tangents_47.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf66.data_ptr()))
    del tangents_47
    buf67 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (2048, 128), (1, 2048), 0), view_486, out=buf67)
    buf68 = reinterpret_tensor(buf60, (128, 2048), (2048, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (128, 2048), (2048, 1), 0), permute_325, out=buf68)
    del permute_325
    buf69 = buf66; del buf66  # reuse
    cpp_fused_clone_15(c_void_p(tangents_46.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf69.data_ptr()))
    del tangents_46
    buf70 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (2048, 128), (1, 2048), 0), view_486, out=buf70)
    buf71 = reinterpret_tensor(buf64, (128, 2048), (2048, 1), 0); del buf64  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0), permute_329, out=buf71)
    del permute_329
    buf72 = reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0); del buf69  # reuse
    cpp_fused_view_16(c_void_p(buf65.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (2048, 128), (1, 2048), 0), view_486, out=buf73)
    del view_486
    buf74 = reinterpret_tensor(buf65, (128, 2048), (2048, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf72, permute_333, out=buf74)
    del permute_333
    buf75 = buf53; del buf53  # reuse
    buf76 = buf52; del buf52  # reuse
    buf77 = reinterpret_tensor(buf62, (2048, ), (1, ), 0); del buf62  # reuse
    buf78 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf79 = buf56; del buf56  # reuse
    cpp_fused_add_native_layer_norm_backward_17(c_void_p(buf79.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(mul_176.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del div_28
    del mul_176
    del primals_289
    buf80 = reinterpret_tensor(buf48, (128, 8192), (8192, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (128, 2048), (2048, 1), 0), permute_335, out=buf80)
    del permute_335
    buf81 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (2048, 128), (1, 2048), 0), view_484, out=buf81)
    del view_484
    buf82 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf83 = reinterpret_tensor(buf80, (1, 128, 8192), (1048576, 8192, 1), 0); del buf80  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_18(c_void_p(buf83.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(tanh_21.data_ptr()), c_void_p(buf82.data_ptr()))
    del addmm_64
    del tanh_21
    buf84 = buf74; del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (128, 8192), (8192, 1), 0), permute_339, out=buf84)
    del permute_339
    buf85 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (8192, 128), (1, 8192), 0), view_482, out=buf85)
    del view_482
    buf86 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf87 = buf76; del buf76  # reuse
    buf88 = buf75; del buf75  # reuse
    buf89 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf90 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf91 = buf79; del buf79  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_19(c_void_p(buf91.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(mul_170.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del div_29
    del mul_170
    del primals_283
    buf92 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0), permute_343, out=buf92)
    del permute_343
    buf93 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (2048, 128), (1, 2048), 0), view_480, out=buf93)
    del view_480
    buf94 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_20(c_void_p(buf91.data_ptr()), c_void_p(buf94.data_ptr()))
    buf95 = reinterpret_tensor(buf71, (16, 128, 128), (16384, 128, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_348, reinterpret_tensor(buf92, (16, 128, 128), (128, 2048, 1), 0), out=buf95)
    del permute_348
    buf96 = reinterpret_tensor(buf68, (16, 128, 128), (16384, 128, 1), 0); del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf92, (16, 128, 128), (128, 2048, 1), 0), permute_349, out=buf96)
    del permute_349
    buf97 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf98 = reinterpret_tensor(buf96, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf96  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_21(c_void_p(buf98.data_ptr()), c_void_p(alias_53.data_ptr()), c_void_p(slice_88.data_ptr()), c_void_p(buf97.data_ptr()))
    del alias_53
    del slice_88
    buf99 = reinterpret_tensor(buf92, (16, 128, 128), (16384, 128, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_350, reinterpret_tensor(buf98, (16, 128, 128), (16384, 128, 1), 0), out=buf99)
    del permute_350
    buf100 = reinterpret_tensor(buf72, (16, 128, 128), (16384, 128, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf98, (16, 128, 128), (16384, 128, 1), 0), permute_351, out=buf100)
    del permute_351
    buf101 = reinterpret_tensor(buf98, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf98  # reuse
    cpp_fused_clone_22(c_void_p(tangents_45.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf101.data_ptr()))
    del tangents_45
    buf102 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (2048, 128), (1, 2048), 0), view_464, out=buf102)
    buf103 = reinterpret_tensor(buf95, (128, 2048), (2048, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (128, 2048), (2048, 1), 0), permute_358, out=buf103)
    del permute_358
    buf104 = buf101; del buf101  # reuse
    cpp_fused_clone_23(c_void_p(tangents_44.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf104.data_ptr()))
    del tangents_44
    buf105 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (2048, 128), (1, 2048), 0), view_464, out=buf105)
    buf106 = reinterpret_tensor(buf99, (128, 2048), (2048, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (128, 2048), (2048, 1), 0), permute_362, out=buf106)
    del permute_362
    buf107 = reinterpret_tensor(buf104, (128, 2048), (2048, 1), 0); del buf104  # reuse
    cpp_fused_view_24(c_void_p(buf100.data_ptr()), c_void_p(buf107.data_ptr()))
    buf108 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (2048, 128), (1, 2048), 0), view_464, out=buf108)
    del view_464
    buf109 = reinterpret_tensor(buf100, (128, 2048), (2048, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf107, permute_366, out=buf109)
    del permute_366
    buf110 = buf88; del buf88  # reuse
    buf111 = buf87; del buf87  # reuse
    buf112 = reinterpret_tensor(buf97, (2048, ), (1, ), 0); del buf97  # reuse
    buf113 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf114 = reinterpret_tensor(buf103, (1, 128, 2048), (262144, 2048, 1), 0); del buf103  # reuse
    cpp_fused_add_native_layer_norm_backward_25(c_void_p(buf114.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(mul_168.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del div_30
    del mul_168
    del primals_276
    buf115 = reinterpret_tensor(buf83, (128, 8192), (8192, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (128, 2048), (2048, 1), 0), permute_368, out=buf115)
    del permute_368
    buf116 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (2048, 128), (1, 2048), 0), view_462, out=buf116)
    del view_462
    buf117 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf118 = reinterpret_tensor(buf115, (1, 128, 8192), (1048576, 8192, 1), 0); del buf115  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_26(c_void_p(buf118.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(addmm_61.data_ptr()), c_void_p(tanh_20.data_ptr()), c_void_p(buf117.data_ptr()))
    del addmm_61
    del tanh_20
    buf119 = reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf118, (128, 8192), (8192, 1), 0), permute_372, out=buf119)
    del permute_372
    buf120 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf118, (8192, 128), (1, 8192), 0), view_460, out=buf120)
    del view_460
    buf121 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf122 = buf111; del buf111  # reuse
    buf123 = buf110; del buf110  # reuse
    buf124 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf125 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf126 = buf114; del buf114  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf126.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(mul_162.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del div_31
    del mul_162
    del primals_270
    buf127 = buf119; del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (128, 2048), (2048, 1), 0), permute_376, out=buf127)
    del permute_376
    buf128 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (2048, 128), (1, 2048), 0), view_458, out=buf128)
    del view_458
    buf129 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf126.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf109, (16, 128, 128), (16384, 128, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_381, reinterpret_tensor(buf127, (16, 128, 128), (128, 2048, 1), 0), out=buf130)
    del permute_381
    buf131 = reinterpret_tensor(buf106, (16, 128, 128), (16384, 128, 1), 0); del buf106  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf127, (16, 128, 128), (128, 2048, 1), 0), permute_382, out=buf131)
    del permute_382
    buf132 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf133 = reinterpret_tensor(buf131, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf131  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_29(c_void_p(buf133.data_ptr()), c_void_p(alias_55.data_ptr()), c_void_p(slice_84.data_ptr()), c_void_p(buf132.data_ptr()))
    del alias_55
    del slice_84
    buf134 = reinterpret_tensor(buf127, (16, 128, 128), (16384, 128, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_383, reinterpret_tensor(buf133, (16, 128, 128), (16384, 128, 1), 0), out=buf134)
    del permute_383
    buf135 = reinterpret_tensor(buf107, (16, 128, 128), (16384, 128, 1), 0); del buf107  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf133, (16, 128, 128), (16384, 128, 1), 0), permute_384, out=buf135)
    del permute_384
    buf136 = reinterpret_tensor(buf133, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf133  # reuse
    cpp_fused_clone_30(c_void_p(tangents_43.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf136.data_ptr()))
    del tangents_43
    buf137 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (2048, 128), (1, 2048), 0), view_442, out=buf137)
    buf138 = reinterpret_tensor(buf130, (128, 2048), (2048, 1), 0); del buf130  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (128, 2048), (2048, 1), 0), permute_391, out=buf138)
    del permute_391
    buf139 = buf136; del buf136  # reuse
    cpp_fused_clone_31(c_void_p(tangents_42.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf139.data_ptr()))
    del tangents_42
    buf140 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (2048, 128), (1, 2048), 0), view_442, out=buf140)
    buf141 = reinterpret_tensor(buf134, (128, 2048), (2048, 1), 0); del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (128, 2048), (2048, 1), 0), permute_395, out=buf141)
    del permute_395
    buf142 = reinterpret_tensor(buf139, (128, 2048), (2048, 1), 0); del buf139  # reuse
    cpp_fused_view_32(c_void_p(buf135.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (2048, 128), (1, 2048), 0), view_442, out=buf143)
    del view_442
    buf144 = reinterpret_tensor(buf135, (128, 2048), (2048, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf142, permute_399, out=buf144)
    del permute_399
    buf145 = buf123; del buf123  # reuse
    buf146 = buf122; del buf122  # reuse
    buf147 = reinterpret_tensor(buf132, (2048, ), (1, ), 0); del buf132  # reuse
    buf148 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf149 = buf126; del buf126  # reuse
    cpp_fused_add_native_layer_norm_backward_33(c_void_p(buf149.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(mul_160.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del div_32
    del mul_160
    del primals_263
    buf150 = reinterpret_tensor(buf118, (128, 8192), (8192, 1), 0); del buf118  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (128, 2048), (2048, 1), 0), permute_401, out=buf150)
    del permute_401
    buf151 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (2048, 128), (1, 2048), 0), view_440, out=buf151)
    del view_440
    buf152 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf150, (1, 128, 8192), (1048576, 8192, 1), 0); del buf150  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_34(c_void_p(buf153.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(tanh_19.data_ptr()), c_void_p(buf152.data_ptr()))
    del addmm_58
    del tanh_19
    buf154 = buf144; del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (128, 8192), (8192, 1), 0), permute_405, out=buf154)
    del permute_405
    buf155 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (8192, 128), (1, 8192), 0), view_438, out=buf155)
    del view_438
    buf156 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf157 = buf146; del buf146  # reuse
    buf158 = buf145; del buf145  # reuse
    buf159 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf160 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf161 = buf149; del buf149  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_35(c_void_p(buf161.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(mul_154.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del div_33
    del mul_154
    del primals_257
    buf162 = buf154; del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (128, 2048), (2048, 1), 0), permute_409, out=buf162)
    del permute_409
    buf163 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (2048, 128), (1, 2048), 0), view_436, out=buf163)
    del view_436
    buf164 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_36(c_void_p(buf161.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf141, (16, 128, 128), (16384, 128, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_414, reinterpret_tensor(buf162, (16, 128, 128), (128, 2048, 1), 0), out=buf165)
    del permute_414
    buf166 = reinterpret_tensor(buf138, (16, 128, 128), (16384, 128, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf162, (16, 128, 128), (128, 2048, 1), 0), permute_415, out=buf166)
    del permute_415
    buf167 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf168 = reinterpret_tensor(buf166, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf166  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_37(c_void_p(buf168.data_ptr()), c_void_p(alias_57.data_ptr()), c_void_p(slice_80.data_ptr()), c_void_p(buf167.data_ptr()))
    del alias_57
    del slice_80
    buf169 = reinterpret_tensor(buf162, (16, 128, 128), (16384, 128, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_416, reinterpret_tensor(buf168, (16, 128, 128), (16384, 128, 1), 0), out=buf169)
    del permute_416
    buf170 = reinterpret_tensor(buf142, (16, 128, 128), (16384, 128, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf168, (16, 128, 128), (16384, 128, 1), 0), permute_417, out=buf170)
    del permute_417
    buf171 = reinterpret_tensor(buf168, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf168  # reuse
    cpp_fused_clone_38(c_void_p(tangents_41.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf171.data_ptr()))
    del tangents_41
    buf172 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (2048, 128), (1, 2048), 0), view_420, out=buf172)
    buf173 = reinterpret_tensor(buf165, (128, 2048), (2048, 1), 0); del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (128, 2048), (2048, 1), 0), permute_424, out=buf173)
    del permute_424
    buf174 = buf171; del buf171  # reuse
    cpp_fused_clone_39(c_void_p(tangents_40.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf174.data_ptr()))
    del tangents_40
    buf175 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (2048, 128), (1, 2048), 0), view_420, out=buf175)
    buf176 = reinterpret_tensor(buf169, (128, 2048), (2048, 1), 0); del buf169  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 2048), (2048, 1), 0), permute_428, out=buf176)
    del permute_428
    buf177 = reinterpret_tensor(buf174, (128, 2048), (2048, 1), 0); del buf174  # reuse
    cpp_fused_view_40(c_void_p(buf170.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (2048, 128), (1, 2048), 0), view_420, out=buf178)
    del view_420
    buf179 = reinterpret_tensor(buf170, (128, 2048), (2048, 1), 0); del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf177, permute_432, out=buf179)
    del permute_432
    buf180 = buf158; del buf158  # reuse
    buf181 = buf157; del buf157  # reuse
    buf182 = reinterpret_tensor(buf167, (2048, ), (1, ), 0); del buf167  # reuse
    buf183 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf184 = buf161; del buf161  # reuse
    cpp_fused_add_native_layer_norm_backward_41(c_void_p(buf184.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(mul_152.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del div_34
    del mul_152
    del primals_250
    buf185 = reinterpret_tensor(buf153, (128, 8192), (8192, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (128, 2048), (2048, 1), 0), permute_434, out=buf185)
    del permute_434
    buf186 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (2048, 128), (1, 2048), 0), view_418, out=buf186)
    del view_418
    buf187 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf188 = reinterpret_tensor(buf185, (1, 128, 8192), (1048576, 8192, 1), 0); del buf185  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_42(c_void_p(buf188.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(addmm_55.data_ptr()), c_void_p(tanh_18.data_ptr()), c_void_p(buf187.data_ptr()))
    del addmm_55
    del tanh_18
    buf189 = buf179; del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (128, 8192), (8192, 1), 0), permute_438, out=buf189)
    del permute_438
    buf190 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (8192, 128), (1, 8192), 0), view_416, out=buf190)
    del view_416
    buf191 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf192 = buf181; del buf181  # reuse
    buf193 = buf180; del buf180  # reuse
    buf194 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf195 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf196 = buf184; del buf184  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_43(c_void_p(buf196.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(mul_146.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del div_35
    del mul_146
    del primals_244
    buf197 = buf189; del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (128, 2048), (2048, 1), 0), permute_442, out=buf197)
    del permute_442
    buf198 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (2048, 128), (1, 2048), 0), view_414, out=buf198)
    del view_414
    buf199 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_44(c_void_p(buf196.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf176, (16, 128, 128), (16384, 128, 1), 0); del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_447, reinterpret_tensor(buf197, (16, 128, 128), (128, 2048, 1), 0), out=buf200)
    del permute_447
    buf201 = reinterpret_tensor(buf173, (16, 128, 128), (16384, 128, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf197, (16, 128, 128), (128, 2048, 1), 0), permute_448, out=buf201)
    del permute_448
    buf202 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf203 = reinterpret_tensor(buf201, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf201  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_45(c_void_p(buf203.data_ptr()), c_void_p(alias_59.data_ptr()), c_void_p(slice_76.data_ptr()), c_void_p(buf202.data_ptr()))
    del alias_59
    del slice_76
    buf204 = reinterpret_tensor(buf197, (16, 128, 128), (16384, 128, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_449, reinterpret_tensor(buf203, (16, 128, 128), (16384, 128, 1), 0), out=buf204)
    del permute_449
    buf205 = reinterpret_tensor(buf177, (16, 128, 128), (16384, 128, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf203, (16, 128, 128), (16384, 128, 1), 0), permute_450, out=buf205)
    del permute_450
    buf206 = reinterpret_tensor(buf203, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf203  # reuse
    cpp_fused_clone_46(c_void_p(tangents_39.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf206.data_ptr()))
    del tangents_39
    buf207 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (2048, 128), (1, 2048), 0), view_398, out=buf207)
    buf208 = reinterpret_tensor(buf200, (128, 2048), (2048, 1), 0); del buf200  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (128, 2048), (2048, 1), 0), permute_457, out=buf208)
    del permute_457
    buf209 = buf206; del buf206  # reuse
    cpp_fused_clone_47(c_void_p(tangents_38.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf209.data_ptr()))
    del tangents_38
    buf210 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (2048, 128), (1, 2048), 0), view_398, out=buf210)
    buf211 = reinterpret_tensor(buf204, (128, 2048), (2048, 1), 0); del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (128, 2048), (2048, 1), 0), permute_461, out=buf211)
    del permute_461
    buf212 = reinterpret_tensor(buf209, (128, 2048), (2048, 1), 0); del buf209  # reuse
    cpp_fused_view_48(c_void_p(buf205.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (2048, 128), (1, 2048), 0), view_398, out=buf213)
    del view_398
    buf214 = reinterpret_tensor(buf205, (128, 2048), (2048, 1), 0); del buf205  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf212, permute_465, out=buf214)
    del permute_465
    buf215 = buf193; del buf193  # reuse
    buf216 = buf192; del buf192  # reuse
    buf217 = reinterpret_tensor(buf202, (2048, ), (1, ), 0); del buf202  # reuse
    buf218 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf219 = buf196; del buf196  # reuse
    cpp_fused_add_native_layer_norm_backward_49(c_void_p(buf219.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(mul_144.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    del div_36
    del mul_144
    del primals_237
    buf220 = reinterpret_tensor(buf188, (128, 8192), (8192, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (128, 2048), (2048, 1), 0), permute_467, out=buf220)
    del permute_467
    buf221 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (2048, 128), (1, 2048), 0), view_396, out=buf221)
    del view_396
    buf222 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf223 = reinterpret_tensor(buf220, (1, 128, 8192), (1048576, 8192, 1), 0); del buf220  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_50(c_void_p(buf223.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(tanh_17.data_ptr()), c_void_p(buf222.data_ptr()))
    del addmm_52
    del tanh_17
    buf224 = buf214; del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (128, 8192), (8192, 1), 0), permute_471, out=buf224)
    del permute_471
    buf225 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (8192, 128), (1, 8192), 0), view_394, out=buf225)
    del view_394
    buf226 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf227 = buf216; del buf216  # reuse
    buf228 = buf215; del buf215  # reuse
    buf229 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf230 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf231 = buf219; del buf219  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_51(c_void_p(buf231.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(mul_138.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    del div_37
    del mul_138
    del primals_231
    buf232 = buf224; del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (128, 2048), (2048, 1), 0), permute_475, out=buf232)
    del permute_475
    buf233 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (2048, 128), (1, 2048), 0), view_392, out=buf233)
    del view_392
    buf234 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_52(c_void_p(buf231.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = reinterpret_tensor(buf211, (16, 128, 128), (16384, 128, 1), 0); del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_480, reinterpret_tensor(buf232, (16, 128, 128), (128, 2048, 1), 0), out=buf235)
    del permute_480
    buf236 = reinterpret_tensor(buf208, (16, 128, 128), (16384, 128, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf232, (16, 128, 128), (128, 2048, 1), 0), permute_481, out=buf236)
    del permute_481
    buf237 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf238 = reinterpret_tensor(buf236, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf236  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_53(c_void_p(buf238.data_ptr()), c_void_p(alias_61.data_ptr()), c_void_p(slice_72.data_ptr()), c_void_p(buf237.data_ptr()))
    del alias_61
    del slice_72
    buf239 = reinterpret_tensor(buf232, (16, 128, 128), (16384, 128, 1), 0); del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_482, reinterpret_tensor(buf238, (16, 128, 128), (16384, 128, 1), 0), out=buf239)
    del permute_482
    buf240 = reinterpret_tensor(buf212, (16, 128, 128), (16384, 128, 1), 0); del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf238, (16, 128, 128), (16384, 128, 1), 0), permute_483, out=buf240)
    del permute_483
    buf241 = reinterpret_tensor(buf238, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf238  # reuse
    cpp_fused_clone_54(c_void_p(tangents_37.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf241.data_ptr()))
    del tangents_37
    buf242 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf241, (2048, 128), (1, 2048), 0), view_376, out=buf242)
    buf243 = reinterpret_tensor(buf235, (128, 2048), (2048, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf241, (128, 2048), (2048, 1), 0), permute_490, out=buf243)
    del permute_490
    buf244 = buf241; del buf241  # reuse
    cpp_fused_clone_55(c_void_p(tangents_36.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf244.data_ptr()))
    del tangents_36
    buf245 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (2048, 128), (1, 2048), 0), view_376, out=buf245)
    buf246 = reinterpret_tensor(buf239, (128, 2048), (2048, 1), 0); del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (128, 2048), (2048, 1), 0), permute_494, out=buf246)
    del permute_494
    buf247 = reinterpret_tensor(buf244, (128, 2048), (2048, 1), 0); del buf244  # reuse
    cpp_fused_view_56(c_void_p(buf240.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (2048, 128), (1, 2048), 0), view_376, out=buf248)
    del view_376
    buf249 = reinterpret_tensor(buf240, (128, 2048), (2048, 1), 0); del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf247, permute_498, out=buf249)
    del permute_498
    buf250 = buf228; del buf228  # reuse
    buf251 = buf227; del buf227  # reuse
    buf252 = reinterpret_tensor(buf237, (2048, ), (1, ), 0); del buf237  # reuse
    buf253 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf254 = buf231; del buf231  # reuse
    cpp_fused_add_native_layer_norm_backward_57(c_void_p(buf254.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del div_38
    del mul_136
    del primals_224
    buf255 = reinterpret_tensor(buf223, (128, 8192), (8192, 1), 0); del buf223  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (128, 2048), (2048, 1), 0), permute_500, out=buf255)
    del permute_500
    buf256 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (2048, 128), (1, 2048), 0), view_374, out=buf256)
    del view_374
    buf257 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf258 = reinterpret_tensor(buf255, (1, 128, 8192), (1048576, 8192, 1), 0); del buf255  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_58(c_void_p(buf258.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(addmm_49.data_ptr()), c_void_p(tanh_16.data_ptr()), c_void_p(buf257.data_ptr()))
    del addmm_49
    del tanh_16
    buf259 = buf249; del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (128, 8192), (8192, 1), 0), permute_504, out=buf259)
    del permute_504
    buf260 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (8192, 128), (1, 8192), 0), view_372, out=buf260)
    del view_372
    buf261 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf262 = buf251; del buf251  # reuse
    buf263 = buf250; del buf250  # reuse
    buf264 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf265 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf266 = buf254; del buf254  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_59(c_void_p(buf266.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(mul_130.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del div_39
    del mul_130
    del primals_218
    buf267 = buf259; del buf259  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (128, 2048), (2048, 1), 0), permute_508, out=buf267)
    del permute_508
    buf268 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (2048, 128), (1, 2048), 0), view_370, out=buf268)
    del view_370
    buf269 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf266.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf246, (16, 128, 128), (16384, 128, 1), 0); del buf246  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_513, reinterpret_tensor(buf267, (16, 128, 128), (128, 2048, 1), 0), out=buf270)
    del permute_513
    buf271 = reinterpret_tensor(buf243, (16, 128, 128), (16384, 128, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf267, (16, 128, 128), (128, 2048, 1), 0), permute_514, out=buf271)
    del permute_514
    buf272 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf273 = reinterpret_tensor(buf271, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf271  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_61(c_void_p(buf273.data_ptr()), c_void_p(alias_63.data_ptr()), c_void_p(slice_68.data_ptr()), c_void_p(buf272.data_ptr()))
    del alias_63
    del slice_68
    buf274 = reinterpret_tensor(buf267, (16, 128, 128), (16384, 128, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_515, reinterpret_tensor(buf273, (16, 128, 128), (16384, 128, 1), 0), out=buf274)
    del permute_515
    buf275 = reinterpret_tensor(buf247, (16, 128, 128), (16384, 128, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf273, (16, 128, 128), (16384, 128, 1), 0), permute_516, out=buf275)
    del permute_516
    buf276 = reinterpret_tensor(buf273, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf273  # reuse
    cpp_fused_clone_62(c_void_p(tangents_35.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf276.data_ptr()))
    del tangents_35
    buf277 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (2048, 128), (1, 2048), 0), view_354, out=buf277)
    buf278 = reinterpret_tensor(buf270, (128, 2048), (2048, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (128, 2048), (2048, 1), 0), permute_523, out=buf278)
    del permute_523
    buf279 = buf276; del buf276  # reuse
    cpp_fused_clone_63(c_void_p(tangents_34.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf279.data_ptr()))
    del tangents_34
    buf280 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (2048, 128), (1, 2048), 0), view_354, out=buf280)
    buf281 = reinterpret_tensor(buf274, (128, 2048), (2048, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (128, 2048), (2048, 1), 0), permute_527, out=buf281)
    del permute_527
    buf282 = reinterpret_tensor(buf279, (128, 2048), (2048, 1), 0); del buf279  # reuse
    cpp_fused_view_64(c_void_p(buf275.data_ptr()), c_void_p(buf282.data_ptr()))
    buf283 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (2048, 128), (1, 2048), 0), view_354, out=buf283)
    del view_354
    buf284 = reinterpret_tensor(buf275, (128, 2048), (2048, 1), 0); del buf275  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf282, permute_531, out=buf284)
    del permute_531
    buf285 = buf263; del buf263  # reuse
    buf286 = buf262; del buf262  # reuse
    buf287 = reinterpret_tensor(buf272, (2048, ), (1, ), 0); del buf272  # reuse
    buf288 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf289 = buf266; del buf266  # reuse
    cpp_fused_add_native_layer_norm_backward_65(c_void_p(buf289.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(mul_128.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del div_40
    del mul_128
    del primals_211
    buf290 = reinterpret_tensor(buf258, (128, 8192), (8192, 1), 0); del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (128, 2048), (2048, 1), 0), permute_533, out=buf290)
    del permute_533
    buf291 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (2048, 128), (1, 2048), 0), view_352, out=buf291)
    del view_352
    buf292 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf293 = reinterpret_tensor(buf290, (1, 128, 8192), (1048576, 8192, 1), 0); del buf290  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_66(c_void_p(buf293.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(tanh_15.data_ptr()), c_void_p(buf292.data_ptr()))
    del addmm_46
    del tanh_15
    buf294 = buf284; del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf293, (128, 8192), (8192, 1), 0), permute_537, out=buf294)
    del permute_537
    buf295 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf293, (8192, 128), (1, 8192), 0), view_350, out=buf295)
    del view_350
    buf296 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf297 = buf286; del buf286  # reuse
    buf298 = buf285; del buf285  # reuse
    buf299 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf300 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf301 = buf289; del buf289  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_67(c_void_p(buf301.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(mul_122.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del div_41
    del mul_122
    del primals_205
    buf302 = buf294; del buf294  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (128, 2048), (2048, 1), 0), permute_541, out=buf302)
    del permute_541
    buf303 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (2048, 128), (1, 2048), 0), view_348, out=buf303)
    del view_348
    buf304 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_68(c_void_p(buf301.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf281, (16, 128, 128), (16384, 128, 1), 0); del buf281  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_546, reinterpret_tensor(buf302, (16, 128, 128), (128, 2048, 1), 0), out=buf305)
    del permute_546
    buf306 = reinterpret_tensor(buf278, (16, 128, 128), (16384, 128, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf302, (16, 128, 128), (128, 2048, 1), 0), permute_547, out=buf306)
    del permute_547
    buf307 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf306, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf306  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_69(c_void_p(buf308.data_ptr()), c_void_p(alias_65.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf307.data_ptr()))
    del alias_65
    del slice_64
    buf309 = reinterpret_tensor(buf302, (16, 128, 128), (16384, 128, 1), 0); del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_548, reinterpret_tensor(buf308, (16, 128, 128), (16384, 128, 1), 0), out=buf309)
    del permute_548
    buf310 = reinterpret_tensor(buf282, (16, 128, 128), (16384, 128, 1), 0); del buf282  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf308, (16, 128, 128), (16384, 128, 1), 0), permute_549, out=buf310)
    del permute_549
    buf311 = reinterpret_tensor(buf308, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf308  # reuse
    cpp_fused_clone_70(c_void_p(tangents_33.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf311.data_ptr()))
    del tangents_33
    buf312 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (2048, 128), (1, 2048), 0), view_332, out=buf312)
    buf313 = reinterpret_tensor(buf305, (128, 2048), (2048, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0), permute_556, out=buf313)
    del permute_556
    buf314 = buf311; del buf311  # reuse
    cpp_fused_clone_71(c_void_p(tangents_32.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf314.data_ptr()))
    del tangents_32
    buf315 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (2048, 128), (1, 2048), 0), view_332, out=buf315)
    buf316 = reinterpret_tensor(buf309, (128, 2048), (2048, 1), 0); del buf309  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (128, 2048), (2048, 1), 0), permute_560, out=buf316)
    del permute_560
    buf317 = reinterpret_tensor(buf314, (128, 2048), (2048, 1), 0); del buf314  # reuse
    cpp_fused_view_72(c_void_p(buf310.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (2048, 128), (1, 2048), 0), view_332, out=buf318)
    del view_332
    buf319 = reinterpret_tensor(buf310, (128, 2048), (2048, 1), 0); del buf310  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf317, permute_564, out=buf319)
    del permute_564
    buf320 = buf298; del buf298  # reuse
    buf321 = buf297; del buf297  # reuse
    buf322 = reinterpret_tensor(buf307, (2048, ), (1, ), 0); del buf307  # reuse
    buf323 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf324 = buf301; del buf301  # reuse
    cpp_fused_add_native_layer_norm_backward_73(c_void_p(buf324.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del div_42
    del mul_120
    del primals_198
    buf325 = reinterpret_tensor(buf293, (128, 8192), (8192, 1), 0); del buf293  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (128, 2048), (2048, 1), 0), permute_566, out=buf325)
    del permute_566
    buf326 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (2048, 128), (1, 2048), 0), view_330, out=buf326)
    del view_330
    buf327 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf328 = reinterpret_tensor(buf325, (1, 128, 8192), (1048576, 8192, 1), 0); del buf325  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_74(c_void_p(buf328.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(addmm_43.data_ptr()), c_void_p(tanh_14.data_ptr()), c_void_p(buf327.data_ptr()))
    del addmm_43
    del tanh_14
    buf329 = buf319; del buf319  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (128, 8192), (8192, 1), 0), permute_570, out=buf329)
    del permute_570
    buf330 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (8192, 128), (1, 8192), 0), view_328, out=buf330)
    del view_328
    buf331 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf332 = buf321; del buf321  # reuse
    buf333 = buf320; del buf320  # reuse
    buf334 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf335 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf336 = buf324; del buf324  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_75(c_void_p(buf336.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(mul_114.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del div_43
    del mul_114
    del primals_192
    buf337 = buf329; del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (128, 2048), (2048, 1), 0), permute_574, out=buf337)
    del permute_574
    buf338 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (2048, 128), (1, 2048), 0), view_326, out=buf338)
    del view_326
    buf339 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_76(c_void_p(buf336.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf316, (16, 128, 128), (16384, 128, 1), 0); del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_579, reinterpret_tensor(buf337, (16, 128, 128), (128, 2048, 1), 0), out=buf340)
    del permute_579
    buf341 = reinterpret_tensor(buf313, (16, 128, 128), (16384, 128, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf337, (16, 128, 128), (128, 2048, 1), 0), permute_580, out=buf341)
    del permute_580
    buf342 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf343 = reinterpret_tensor(buf341, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf341  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_77(c_void_p(buf343.data_ptr()), c_void_p(alias_67.data_ptr()), c_void_p(slice_60.data_ptr()), c_void_p(buf342.data_ptr()))
    del alias_67
    del slice_60
    buf344 = reinterpret_tensor(buf337, (16, 128, 128), (16384, 128, 1), 0); del buf337  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_581, reinterpret_tensor(buf343, (16, 128, 128), (16384, 128, 1), 0), out=buf344)
    del permute_581
    buf345 = reinterpret_tensor(buf317, (16, 128, 128), (16384, 128, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf343, (16, 128, 128), (16384, 128, 1), 0), permute_582, out=buf345)
    del permute_582
    buf346 = reinterpret_tensor(buf343, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf343  # reuse
    cpp_fused_clone_78(c_void_p(tangents_31.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf346.data_ptr()))
    del tangents_31
    buf347 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf346, (2048, 128), (1, 2048), 0), view_310, out=buf347)
    buf348 = reinterpret_tensor(buf340, (128, 2048), (2048, 1), 0); del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf346, (128, 2048), (2048, 1), 0), permute_589, out=buf348)
    del permute_589
    buf349 = buf346; del buf346  # reuse
    cpp_fused_clone_79(c_void_p(tangents_30.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf349.data_ptr()))
    del tangents_30
    buf350 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (2048, 128), (1, 2048), 0), view_310, out=buf350)
    buf351 = reinterpret_tensor(buf344, (128, 2048), (2048, 1), 0); del buf344  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (128, 2048), (2048, 1), 0), permute_593, out=buf351)
    del permute_593
    buf352 = reinterpret_tensor(buf349, (128, 2048), (2048, 1), 0); del buf349  # reuse
    cpp_fused_view_80(c_void_p(buf345.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (2048, 128), (1, 2048), 0), view_310, out=buf353)
    del view_310
    buf354 = reinterpret_tensor(buf345, (128, 2048), (2048, 1), 0); del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf352, permute_597, out=buf354)
    del permute_597
    buf355 = buf333; del buf333  # reuse
    buf356 = buf332; del buf332  # reuse
    buf357 = reinterpret_tensor(buf342, (2048, ), (1, ), 0); del buf342  # reuse
    buf358 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf359 = buf336; del buf336  # reuse
    cpp_fused_add_native_layer_norm_backward_81(c_void_p(buf359.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(mul_112.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del div_44
    del mul_112
    del primals_185
    buf360 = reinterpret_tensor(buf328, (128, 8192), (8192, 1), 0); del buf328  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (128, 2048), (2048, 1), 0), permute_599, out=buf360)
    del permute_599
    buf361 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (2048, 128), (1, 2048), 0), view_308, out=buf361)
    del view_308
    buf362 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf360, (1, 128, 8192), (1048576, 8192, 1), 0); del buf360  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_82(c_void_p(buf363.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(tanh_13.data_ptr()), c_void_p(buf362.data_ptr()))
    del addmm_40
    del tanh_13
    buf364 = buf354; del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (128, 8192), (8192, 1), 0), permute_603, out=buf364)
    del permute_603
    buf365 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (8192, 128), (1, 8192), 0), view_306, out=buf365)
    del view_306
    buf366 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf367 = buf356; del buf356  # reuse
    buf368 = buf355; del buf355  # reuse
    buf369 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf370 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf371 = buf359; del buf359  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_83(c_void_p(buf371.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(mul_106.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del div_45
    del mul_106
    del primals_179
    buf372 = buf364; del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (128, 2048), (2048, 1), 0), permute_607, out=buf372)
    del permute_607
    buf373 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (2048, 128), (1, 2048), 0), view_304, out=buf373)
    del view_304
    buf374 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf371.data_ptr()), c_void_p(buf374.data_ptr()))
    buf375 = reinterpret_tensor(buf351, (16, 128, 128), (16384, 128, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_612, reinterpret_tensor(buf372, (16, 128, 128), (128, 2048, 1), 0), out=buf375)
    del permute_612
    buf376 = reinterpret_tensor(buf348, (16, 128, 128), (16384, 128, 1), 0); del buf348  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf372, (16, 128, 128), (128, 2048, 1), 0), permute_613, out=buf376)
    del permute_613
    buf377 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf378 = reinterpret_tensor(buf376, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf376  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_85(c_void_p(buf378.data_ptr()), c_void_p(alias_69.data_ptr()), c_void_p(slice_56.data_ptr()), c_void_p(buf377.data_ptr()))
    del alias_69
    del slice_56
    buf379 = reinterpret_tensor(buf372, (16, 128, 128), (16384, 128, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_614, reinterpret_tensor(buf378, (16, 128, 128), (16384, 128, 1), 0), out=buf379)
    del permute_614
    buf380 = reinterpret_tensor(buf352, (16, 128, 128), (16384, 128, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf378, (16, 128, 128), (16384, 128, 1), 0), permute_615, out=buf380)
    del permute_615
    buf381 = reinterpret_tensor(buf378, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf378  # reuse
    cpp_fused_clone_86(c_void_p(tangents_29.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf381.data_ptr()))
    del tangents_29
    buf382 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (2048, 128), (1, 2048), 0), view_288, out=buf382)
    buf383 = reinterpret_tensor(buf375, (128, 2048), (2048, 1), 0); del buf375  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (128, 2048), (2048, 1), 0), permute_622, out=buf383)
    del permute_622
    buf384 = buf381; del buf381  # reuse
    cpp_fused_clone_87(c_void_p(tangents_28.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf384.data_ptr()))
    del tangents_28
    buf385 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (2048, 128), (1, 2048), 0), view_288, out=buf385)
    buf386 = reinterpret_tensor(buf379, (128, 2048), (2048, 1), 0); del buf379  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (128, 2048), (2048, 1), 0), permute_626, out=buf386)
    del permute_626
    buf387 = reinterpret_tensor(buf384, (128, 2048), (2048, 1), 0); del buf384  # reuse
    cpp_fused_view_88(c_void_p(buf380.data_ptr()), c_void_p(buf387.data_ptr()))
    buf388 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (2048, 128), (1, 2048), 0), view_288, out=buf388)
    del view_288
    buf389 = reinterpret_tensor(buf380, (128, 2048), (2048, 1), 0); del buf380  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, permute_630, out=buf389)
    del permute_630
    buf390 = buf368; del buf368  # reuse
    buf391 = buf367; del buf367  # reuse
    buf392 = reinterpret_tensor(buf377, (2048, ), (1, ), 0); del buf377  # reuse
    buf393 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf394 = buf371; del buf371  # reuse
    cpp_fused_add_native_layer_norm_backward_89(c_void_p(buf394.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(mul_104.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    del div_46
    del mul_104
    del primals_172
    buf395 = reinterpret_tensor(buf363, (128, 8192), (8192, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (128, 2048), (2048, 1), 0), permute_632, out=buf395)
    del permute_632
    buf396 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (2048, 128), (1, 2048), 0), view_286, out=buf396)
    del view_286
    buf397 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf398 = reinterpret_tensor(buf395, (1, 128, 8192), (1048576, 8192, 1), 0); del buf395  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_90(c_void_p(buf398.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(addmm_37.data_ptr()), c_void_p(tanh_12.data_ptr()), c_void_p(buf397.data_ptr()))
    del addmm_37
    del tanh_12
    buf399 = buf389; del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf398, (128, 8192), (8192, 1), 0), permute_636, out=buf399)
    del permute_636
    buf400 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf398, (8192, 128), (1, 8192), 0), view_284, out=buf400)
    del view_284
    buf401 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf402 = buf391; del buf391  # reuse
    buf403 = buf390; del buf390  # reuse
    buf404 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf405 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf406 = buf394; del buf394  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_91(c_void_p(buf406.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(mul_98.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del div_47
    del mul_98
    del primals_166
    buf407 = buf399; del buf399  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (128, 2048), (2048, 1), 0), permute_640, out=buf407)
    del permute_640
    buf408 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (2048, 128), (1, 2048), 0), view_282, out=buf408)
    del view_282
    buf409 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_92(c_void_p(buf406.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf386, (16, 128, 128), (16384, 128, 1), 0); del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_645, reinterpret_tensor(buf407, (16, 128, 128), (128, 2048, 1), 0), out=buf410)
    del permute_645
    buf411 = reinterpret_tensor(buf383, (16, 128, 128), (16384, 128, 1), 0); del buf383  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf407, (16, 128, 128), (128, 2048, 1), 0), permute_646, out=buf411)
    del permute_646
    buf412 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf411, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf411  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_93(c_void_p(buf413.data_ptr()), c_void_p(alias_71.data_ptr()), c_void_p(slice_52.data_ptr()), c_void_p(buf412.data_ptr()))
    del alias_71
    del slice_52
    buf414 = reinterpret_tensor(buf407, (16, 128, 128), (16384, 128, 1), 0); del buf407  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_647, reinterpret_tensor(buf413, (16, 128, 128), (16384, 128, 1), 0), out=buf414)
    del permute_647
    buf415 = reinterpret_tensor(buf387, (16, 128, 128), (16384, 128, 1), 0); del buf387  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf413, (16, 128, 128), (16384, 128, 1), 0), permute_648, out=buf415)
    del permute_648
    buf416 = reinterpret_tensor(buf413, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf413  # reuse
    cpp_fused_clone_94(c_void_p(tangents_27.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf416.data_ptr()))
    del tangents_27
    buf417 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (2048, 128), (1, 2048), 0), view_266, out=buf417)
    buf418 = reinterpret_tensor(buf410, (128, 2048), (2048, 1), 0); del buf410  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (128, 2048), (2048, 1), 0), permute_655, out=buf418)
    del permute_655
    buf419 = buf416; del buf416  # reuse
    cpp_fused_clone_95(c_void_p(tangents_26.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf419.data_ptr()))
    del tangents_26
    buf420 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (2048, 128), (1, 2048), 0), view_266, out=buf420)
    buf421 = reinterpret_tensor(buf414, (128, 2048), (2048, 1), 0); del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (128, 2048), (2048, 1), 0), permute_659, out=buf421)
    del permute_659
    buf422 = reinterpret_tensor(buf419, (128, 2048), (2048, 1), 0); del buf419  # reuse
    cpp_fused_view_96(c_void_p(buf415.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (2048, 128), (1, 2048), 0), view_266, out=buf423)
    del view_266
    buf424 = reinterpret_tensor(buf415, (128, 2048), (2048, 1), 0); del buf415  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, permute_663, out=buf424)
    del permute_663
    buf425 = buf403; del buf403  # reuse
    buf426 = buf402; del buf402  # reuse
    buf427 = reinterpret_tensor(buf412, (2048, ), (1, ), 0); del buf412  # reuse
    buf428 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf429 = buf406; del buf406  # reuse
    cpp_fused_add_native_layer_norm_backward_97(c_void_p(buf429.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del div_48
    del mul_96
    del primals_159
    buf430 = reinterpret_tensor(buf398, (128, 8192), (8192, 1), 0); del buf398  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf429, (128, 2048), (2048, 1), 0), permute_665, out=buf430)
    del permute_665
    buf431 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf429, (2048, 128), (1, 2048), 0), view_264, out=buf431)
    del view_264
    buf432 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf433 = reinterpret_tensor(buf430, (1, 128, 8192), (1048576, 8192, 1), 0); del buf430  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_98(c_void_p(buf433.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(buf432.data_ptr()))
    del addmm_34
    del tanh_11
    buf434 = buf424; del buf424  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (128, 8192), (8192, 1), 0), permute_669, out=buf434)
    del permute_669
    buf435 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (8192, 128), (1, 8192), 0), view_262, out=buf435)
    del view_262
    buf436 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf437 = buf426; del buf426  # reuse
    buf438 = buf425; del buf425  # reuse
    buf439 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf440 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf441 = buf429; del buf429  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_99(c_void_p(buf441.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()))
    del div_49
    del mul_90
    del primals_153
    buf442 = buf434; del buf434  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf441, (128, 2048), (2048, 1), 0), permute_673, out=buf442)
    del permute_673
    buf443 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf441, (2048, 128), (1, 2048), 0), view_260, out=buf443)
    del view_260
    buf444 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_100(c_void_p(buf441.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = reinterpret_tensor(buf421, (16, 128, 128), (16384, 128, 1), 0); del buf421  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_678, reinterpret_tensor(buf442, (16, 128, 128), (128, 2048, 1), 0), out=buf445)
    del permute_678
    buf446 = reinterpret_tensor(buf418, (16, 128, 128), (16384, 128, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf442, (16, 128, 128), (128, 2048, 1), 0), permute_679, out=buf446)
    del permute_679
    buf447 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf448 = reinterpret_tensor(buf446, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf446  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_101(c_void_p(buf448.data_ptr()), c_void_p(alias_73.data_ptr()), c_void_p(slice_48.data_ptr()), c_void_p(buf447.data_ptr()))
    del alias_73
    del slice_48
    buf449 = reinterpret_tensor(buf442, (16, 128, 128), (16384, 128, 1), 0); del buf442  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_680, reinterpret_tensor(buf448, (16, 128, 128), (16384, 128, 1), 0), out=buf449)
    del permute_680
    buf450 = reinterpret_tensor(buf422, (16, 128, 128), (16384, 128, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf448, (16, 128, 128), (16384, 128, 1), 0), permute_681, out=buf450)
    del permute_681
    buf451 = reinterpret_tensor(buf448, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf448  # reuse
    cpp_fused_clone_102(c_void_p(tangents_25.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf451.data_ptr()))
    del tangents_25
    buf452 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf451, (2048, 128), (1, 2048), 0), view_244, out=buf452)
    buf453 = reinterpret_tensor(buf445, (128, 2048), (2048, 1), 0); del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf451, (128, 2048), (2048, 1), 0), permute_688, out=buf453)
    del permute_688
    buf454 = buf451; del buf451  # reuse
    cpp_fused_clone_103(c_void_p(tangents_24.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf454.data_ptr()))
    del tangents_24
    buf455 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (2048, 128), (1, 2048), 0), view_244, out=buf455)
    buf456 = reinterpret_tensor(buf449, (128, 2048), (2048, 1), 0); del buf449  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (128, 2048), (2048, 1), 0), permute_692, out=buf456)
    del permute_692
    buf457 = reinterpret_tensor(buf454, (128, 2048), (2048, 1), 0); del buf454  # reuse
    cpp_fused_view_104(c_void_p(buf450.data_ptr()), c_void_p(buf457.data_ptr()))
    buf458 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (2048, 128), (1, 2048), 0), view_244, out=buf458)
    del view_244
    buf459 = reinterpret_tensor(buf450, (128, 2048), (2048, 1), 0); del buf450  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf457, permute_696, out=buf459)
    del permute_696
    buf460 = buf438; del buf438  # reuse
    buf461 = buf437; del buf437  # reuse
    buf462 = reinterpret_tensor(buf447, (2048, ), (1, ), 0); del buf447  # reuse
    buf463 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf464 = buf441; del buf441  # reuse
    cpp_fused_add_native_layer_norm_backward_105(c_void_p(buf464.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    del div_50
    del mul_88
    del primals_146
    buf465 = reinterpret_tensor(buf433, (128, 8192), (8192, 1), 0); del buf433  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf464, (128, 2048), (2048, 1), 0), permute_698, out=buf465)
    del permute_698
    buf466 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf464, (2048, 128), (1, 2048), 0), view_242, out=buf466)
    del view_242
    buf467 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf468 = reinterpret_tensor(buf465, (1, 128, 8192), (1048576, 8192, 1), 0); del buf465  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_106(c_void_p(buf468.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(buf467.data_ptr()))
    del addmm_31
    del tanh_10
    buf469 = buf459; del buf459  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (128, 8192), (8192, 1), 0), permute_702, out=buf469)
    del permute_702
    buf470 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (8192, 128), (1, 8192), 0), view_240, out=buf470)
    del view_240
    buf471 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf472 = buf461; del buf461  # reuse
    buf473 = buf460; del buf460  # reuse
    buf474 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf475 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf476 = buf464; del buf464  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_107(c_void_p(buf476.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()))
    del div_51
    del mul_82
    del primals_140
    buf477 = buf469; del buf469  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (128, 2048), (2048, 1), 0), permute_706, out=buf477)
    del permute_706
    buf478 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (2048, 128), (1, 2048), 0), view_238, out=buf478)
    del view_238
    buf479 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_108(c_void_p(buf476.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf456, (16, 128, 128), (16384, 128, 1), 0); del buf456  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_711, reinterpret_tensor(buf477, (16, 128, 128), (128, 2048, 1), 0), out=buf480)
    del permute_711
    buf481 = reinterpret_tensor(buf453, (16, 128, 128), (16384, 128, 1), 0); del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf477, (16, 128, 128), (128, 2048, 1), 0), permute_712, out=buf481)
    del permute_712
    buf482 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf483 = reinterpret_tensor(buf481, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf481  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_109(c_void_p(buf483.data_ptr()), c_void_p(alias_75.data_ptr()), c_void_p(slice_44.data_ptr()), c_void_p(buf482.data_ptr()))
    del alias_75
    del slice_44
    buf484 = reinterpret_tensor(buf477, (16, 128, 128), (16384, 128, 1), 0); del buf477  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_713, reinterpret_tensor(buf483, (16, 128, 128), (16384, 128, 1), 0), out=buf484)
    del permute_713
    buf485 = reinterpret_tensor(buf457, (16, 128, 128), (16384, 128, 1), 0); del buf457  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf483, (16, 128, 128), (16384, 128, 1), 0), permute_714, out=buf485)
    del permute_714
    buf486 = reinterpret_tensor(buf483, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf483  # reuse
    cpp_fused_clone_110(c_void_p(tangents_23.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf486.data_ptr()))
    del tangents_23
    buf487 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (2048, 128), (1, 2048), 0), view_222, out=buf487)
    buf488 = reinterpret_tensor(buf480, (128, 2048), (2048, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (128, 2048), (2048, 1), 0), permute_721, out=buf488)
    del permute_721
    buf489 = buf486; del buf486  # reuse
    cpp_fused_clone_111(c_void_p(tangents_22.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf489.data_ptr()))
    del tangents_22
    buf490 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (2048, 128), (1, 2048), 0), view_222, out=buf490)
    buf491 = reinterpret_tensor(buf484, (128, 2048), (2048, 1), 0); del buf484  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (128, 2048), (2048, 1), 0), permute_725, out=buf491)
    del permute_725
    buf492 = reinterpret_tensor(buf489, (128, 2048), (2048, 1), 0); del buf489  # reuse
    cpp_fused_view_112(c_void_p(buf485.data_ptr()), c_void_p(buf492.data_ptr()))
    buf493 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf492, (2048, 128), (1, 2048), 0), view_222, out=buf493)
    del view_222
    buf494 = reinterpret_tensor(buf485, (128, 2048), (2048, 1), 0); del buf485  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf492, permute_729, out=buf494)
    del permute_729
    buf495 = buf473; del buf473  # reuse
    buf496 = buf472; del buf472  # reuse
    buf497 = reinterpret_tensor(buf482, (2048, ), (1, ), 0); del buf482  # reuse
    buf498 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf499 = buf476; del buf476  # reuse
    cpp_fused_add_native_layer_norm_backward_113(c_void_p(buf499.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    del div_52
    del mul_80
    del primals_133
    buf500 = reinterpret_tensor(buf468, (128, 8192), (8192, 1), 0); del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (128, 2048), (2048, 1), 0), permute_731, out=buf500)
    del permute_731
    buf501 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (2048, 128), (1, 2048), 0), view_220, out=buf501)
    del view_220
    buf502 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf503 = reinterpret_tensor(buf500, (1, 128, 8192), (1048576, 8192, 1), 0); del buf500  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_114(c_void_p(buf503.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(buf502.data_ptr()))
    del addmm_28
    del tanh_9
    buf504 = buf494; del buf494  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (128, 8192), (8192, 1), 0), permute_735, out=buf504)
    del permute_735
    buf505 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (8192, 128), (1, 8192), 0), view_218, out=buf505)
    del view_218
    buf506 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf507 = buf496; del buf496  # reuse
    buf508 = buf495; del buf495  # reuse
    buf509 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf510 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf511 = buf499; del buf499  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_115(c_void_p(buf511.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_74.data_ptr()), c_void_p(div_53.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    del div_53
    del mul_74
    del primals_127
    buf512 = buf504; del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (128, 2048), (2048, 1), 0), permute_739, out=buf512)
    del permute_739
    buf513 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (2048, 128), (1, 2048), 0), view_216, out=buf513)
    del view_216
    buf514 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_116(c_void_p(buf511.data_ptr()), c_void_p(buf514.data_ptr()))
    buf515 = reinterpret_tensor(buf491, (16, 128, 128), (16384, 128, 1), 0); del buf491  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_744, reinterpret_tensor(buf512, (16, 128, 128), (128, 2048, 1), 0), out=buf515)
    del permute_744
    buf516 = reinterpret_tensor(buf488, (16, 128, 128), (16384, 128, 1), 0); del buf488  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf512, (16, 128, 128), (128, 2048, 1), 0), permute_745, out=buf516)
    del permute_745
    buf517 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf518 = reinterpret_tensor(buf516, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf516  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_117(c_void_p(buf518.data_ptr()), c_void_p(alias_77.data_ptr()), c_void_p(slice_40.data_ptr()), c_void_p(buf517.data_ptr()))
    del alias_77
    del slice_40
    buf519 = reinterpret_tensor(buf512, (16, 128, 128), (16384, 128, 1), 0); del buf512  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_746, reinterpret_tensor(buf518, (16, 128, 128), (16384, 128, 1), 0), out=buf519)
    del permute_746
    buf520 = reinterpret_tensor(buf492, (16, 128, 128), (16384, 128, 1), 0); del buf492  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf518, (16, 128, 128), (16384, 128, 1), 0), permute_747, out=buf520)
    del permute_747
    buf521 = reinterpret_tensor(buf518, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf518  # reuse
    cpp_fused_clone_118(c_void_p(tangents_21.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf521.data_ptr()))
    del tangents_21
    buf522 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (2048, 128), (1, 2048), 0), view_200, out=buf522)
    buf523 = reinterpret_tensor(buf515, (128, 2048), (2048, 1), 0); del buf515  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (128, 2048), (2048, 1), 0), permute_754, out=buf523)
    del permute_754
    buf524 = buf521; del buf521  # reuse
    cpp_fused_clone_119(c_void_p(tangents_20.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf524.data_ptr()))
    del tangents_20
    buf525 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (2048, 128), (1, 2048), 0), view_200, out=buf525)
    buf526 = reinterpret_tensor(buf519, (128, 2048), (2048, 1), 0); del buf519  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0), permute_758, out=buf526)
    del permute_758
    buf527 = reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0); del buf524  # reuse
    cpp_fused_view_120(c_void_p(buf520.data_ptr()), c_void_p(buf527.data_ptr()))
    buf528 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (2048, 128), (1, 2048), 0), view_200, out=buf528)
    del view_200
    buf529 = reinterpret_tensor(buf520, (128, 2048), (2048, 1), 0); del buf520  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf527, permute_762, out=buf529)
    del permute_762
    buf530 = buf508; del buf508  # reuse
    buf531 = buf507; del buf507  # reuse
    buf532 = reinterpret_tensor(buf517, (2048, ), (1, ), 0); del buf517  # reuse
    buf533 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf534 = buf511; del buf511  # reuse
    cpp_fused_add_native_layer_norm_backward_121(c_void_p(buf534.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()))
    del div_54
    del mul_72
    del primals_120
    buf535 = reinterpret_tensor(buf503, (128, 8192), (8192, 1), 0); del buf503  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf534, (128, 2048), (2048, 1), 0), permute_764, out=buf535)
    del permute_764
    buf536 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf534, (2048, 128), (1, 2048), 0), view_198, out=buf536)
    del view_198
    buf537 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf538 = reinterpret_tensor(buf535, (1, 128, 8192), (1048576, 8192, 1), 0); del buf535  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_122(c_void_p(buf538.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(buf537.data_ptr()))
    del addmm_25
    del tanh_8
    buf539 = buf529; del buf529  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf538, (128, 8192), (8192, 1), 0), permute_768, out=buf539)
    del permute_768
    buf540 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf538, (8192, 128), (1, 8192), 0), view_196, out=buf540)
    del view_196
    buf541 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf542 = buf531; del buf531  # reuse
    buf543 = buf530; del buf530  # reuse
    buf544 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf545 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf546 = buf534; del buf534  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_123(c_void_p(buf546.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    del div_55
    del mul_66
    del primals_114
    buf547 = buf539; del buf539  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf546, (128, 2048), (2048, 1), 0), permute_772, out=buf547)
    del permute_772
    buf548 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf546, (2048, 128), (1, 2048), 0), view_194, out=buf548)
    del view_194
    buf549 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_124(c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()))
    buf550 = reinterpret_tensor(buf526, (16, 128, 128), (16384, 128, 1), 0); del buf526  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_777, reinterpret_tensor(buf547, (16, 128, 128), (128, 2048, 1), 0), out=buf550)
    del permute_777
    buf551 = reinterpret_tensor(buf523, (16, 128, 128), (16384, 128, 1), 0); del buf523  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf547, (16, 128, 128), (128, 2048, 1), 0), permute_778, out=buf551)
    del permute_778
    buf552 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf553 = reinterpret_tensor(buf551, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf551  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_125(c_void_p(buf553.data_ptr()), c_void_p(alias_79.data_ptr()), c_void_p(slice_36.data_ptr()), c_void_p(buf552.data_ptr()))
    del alias_79
    del slice_36
    buf554 = reinterpret_tensor(buf547, (16, 128, 128), (16384, 128, 1), 0); del buf547  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_779, reinterpret_tensor(buf553, (16, 128, 128), (16384, 128, 1), 0), out=buf554)
    del permute_779
    buf555 = reinterpret_tensor(buf527, (16, 128, 128), (16384, 128, 1), 0); del buf527  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf553, (16, 128, 128), (16384, 128, 1), 0), permute_780, out=buf555)
    del permute_780
    buf556 = reinterpret_tensor(buf553, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf553  # reuse
    cpp_fused_clone_126(c_void_p(tangents_19.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf556.data_ptr()))
    del tangents_19
    buf557 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf556, (2048, 128), (1, 2048), 0), view_178, out=buf557)
    buf558 = reinterpret_tensor(buf550, (128, 2048), (2048, 1), 0); del buf550  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf556, (128, 2048), (2048, 1), 0), permute_787, out=buf558)
    del permute_787
    buf559 = buf556; del buf556  # reuse
    cpp_fused_clone_127(c_void_p(tangents_18.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf559.data_ptr()))
    del tangents_18
    buf560 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf559, (2048, 128), (1, 2048), 0), view_178, out=buf560)
    buf561 = reinterpret_tensor(buf554, (128, 2048), (2048, 1), 0); del buf554  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf559, (128, 2048), (2048, 1), 0), permute_791, out=buf561)
    del permute_791
    buf562 = reinterpret_tensor(buf559, (128, 2048), (2048, 1), 0); del buf559  # reuse
    cpp_fused_view_128(c_void_p(buf555.data_ptr()), c_void_p(buf562.data_ptr()))
    buf563 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf562, (2048, 128), (1, 2048), 0), view_178, out=buf563)
    del view_178
    buf564 = reinterpret_tensor(buf555, (128, 2048), (2048, 1), 0); del buf555  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf562, permute_795, out=buf564)
    del permute_795
    buf565 = buf543; del buf543  # reuse
    buf566 = buf542; del buf542  # reuse
    buf567 = reinterpret_tensor(buf552, (2048, ), (1, ), 0); del buf552  # reuse
    buf568 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf569 = buf546; del buf546  # reuse
    cpp_fused_add_native_layer_norm_backward_129(c_void_p(buf569.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    del div_56
    del mul_64
    del primals_107
    buf570 = reinterpret_tensor(buf538, (128, 8192), (8192, 1), 0); del buf538  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf569, (128, 2048), (2048, 1), 0), permute_797, out=buf570)
    del permute_797
    buf571 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf569, (2048, 128), (1, 2048), 0), view_176, out=buf571)
    del view_176
    buf572 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf573 = reinterpret_tensor(buf570, (1, 128, 8192), (1048576, 8192, 1), 0); del buf570  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_130(c_void_p(buf573.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(buf572.data_ptr()))
    del addmm_22
    del tanh_7
    buf574 = buf564; del buf564  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf573, (128, 8192), (8192, 1), 0), permute_801, out=buf574)
    del permute_801
    buf575 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf573, (8192, 128), (1, 8192), 0), view_174, out=buf575)
    del view_174
    buf576 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf577 = buf566; del buf566  # reuse
    buf578 = buf565; del buf565  # reuse
    buf579 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf580 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf581 = buf569; del buf569  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_131(c_void_p(buf581.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()))
    del div_57
    del mul_58
    del primals_101
    buf582 = buf574; del buf574  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (128, 2048), (2048, 1), 0), permute_805, out=buf582)
    del permute_805
    buf583 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (2048, 128), (1, 2048), 0), view_172, out=buf583)
    del view_172
    buf584 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_132(c_void_p(buf581.data_ptr()), c_void_p(buf584.data_ptr()))
    buf585 = reinterpret_tensor(buf561, (16, 128, 128), (16384, 128, 1), 0); del buf561  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_810, reinterpret_tensor(buf582, (16, 128, 128), (128, 2048, 1), 0), out=buf585)
    del permute_810
    buf586 = reinterpret_tensor(buf558, (16, 128, 128), (16384, 128, 1), 0); del buf558  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf582, (16, 128, 128), (128, 2048, 1), 0), permute_811, out=buf586)
    del permute_811
    buf587 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf588 = reinterpret_tensor(buf586, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf586  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_133(c_void_p(buf588.data_ptr()), c_void_p(alias_81.data_ptr()), c_void_p(slice_32.data_ptr()), c_void_p(buf587.data_ptr()))
    del alias_81
    del slice_32
    buf589 = reinterpret_tensor(buf582, (16, 128, 128), (16384, 128, 1), 0); del buf582  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_812, reinterpret_tensor(buf588, (16, 128, 128), (16384, 128, 1), 0), out=buf589)
    del permute_812
    buf590 = reinterpret_tensor(buf562, (16, 128, 128), (16384, 128, 1), 0); del buf562  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf588, (16, 128, 128), (16384, 128, 1), 0), permute_813, out=buf590)
    del permute_813
    buf591 = reinterpret_tensor(buf588, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf588  # reuse
    cpp_fused_clone_134(c_void_p(tangents_17.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf591.data_ptr()))
    del tangents_17
    buf592 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf591, (2048, 128), (1, 2048), 0), view_156, out=buf592)
    buf593 = reinterpret_tensor(buf585, (128, 2048), (2048, 1), 0); del buf585  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf591, (128, 2048), (2048, 1), 0), permute_820, out=buf593)
    del permute_820
    buf594 = buf591; del buf591  # reuse
    cpp_fused_clone_135(c_void_p(tangents_16.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf594.data_ptr()))
    del tangents_16
    buf595 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf594, (2048, 128), (1, 2048), 0), view_156, out=buf595)
    buf596 = reinterpret_tensor(buf589, (128, 2048), (2048, 1), 0); del buf589  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf594, (128, 2048), (2048, 1), 0), permute_824, out=buf596)
    del permute_824
    buf597 = reinterpret_tensor(buf594, (128, 2048), (2048, 1), 0); del buf594  # reuse
    cpp_fused_view_136(c_void_p(buf590.data_ptr()), c_void_p(buf597.data_ptr()))
    buf598 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf597, (2048, 128), (1, 2048), 0), view_156, out=buf598)
    del view_156
    buf599 = reinterpret_tensor(buf590, (128, 2048), (2048, 1), 0); del buf590  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf597, permute_828, out=buf599)
    del permute_828
    buf600 = buf578; del buf578  # reuse
    buf601 = buf577; del buf577  # reuse
    buf602 = reinterpret_tensor(buf587, (2048, ), (1, ), 0); del buf587  # reuse
    buf603 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf604 = buf581; del buf581  # reuse
    cpp_fused_add_native_layer_norm_backward_137(c_void_p(buf604.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()))
    del div_58
    del mul_56
    del primals_94
    buf605 = reinterpret_tensor(buf573, (128, 8192), (8192, 1), 0); del buf573  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf604, (128, 2048), (2048, 1), 0), permute_830, out=buf605)
    del permute_830
    buf606 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf604, (2048, 128), (1, 2048), 0), view_154, out=buf606)
    del view_154
    buf607 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf608 = reinterpret_tensor(buf605, (1, 128, 8192), (1048576, 8192, 1), 0); del buf605  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_138(c_void_p(buf608.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(buf607.data_ptr()))
    del addmm_19
    del tanh_6
    buf609 = buf599; del buf599  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (128, 8192), (8192, 1), 0), permute_834, out=buf609)
    del permute_834
    buf610 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (8192, 128), (1, 8192), 0), view_152, out=buf610)
    del view_152
    buf611 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf612 = buf601; del buf601  # reuse
    buf613 = buf600; del buf600  # reuse
    buf614 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf615 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf616 = buf604; del buf604  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_139(c_void_p(buf616.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_59.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()))
    del div_59
    del mul_50
    del primals_88
    buf617 = buf609; del buf609  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (128, 2048), (2048, 1), 0), permute_838, out=buf617)
    del permute_838
    buf618 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (2048, 128), (1, 2048), 0), view_150, out=buf618)
    del view_150
    buf619 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_140(c_void_p(buf616.data_ptr()), c_void_p(buf619.data_ptr()))
    buf620 = reinterpret_tensor(buf596, (16, 128, 128), (16384, 128, 1), 0); del buf596  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_843, reinterpret_tensor(buf617, (16, 128, 128), (128, 2048, 1), 0), out=buf620)
    del permute_843
    buf621 = reinterpret_tensor(buf593, (16, 128, 128), (16384, 128, 1), 0); del buf593  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf617, (16, 128, 128), (128, 2048, 1), 0), permute_844, out=buf621)
    del permute_844
    buf622 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf623 = reinterpret_tensor(buf621, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf621  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_141(c_void_p(buf623.data_ptr()), c_void_p(alias_83.data_ptr()), c_void_p(slice_28.data_ptr()), c_void_p(buf622.data_ptr()))
    del alias_83
    del slice_28
    buf624 = reinterpret_tensor(buf617, (16, 128, 128), (16384, 128, 1), 0); del buf617  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_845, reinterpret_tensor(buf623, (16, 128, 128), (16384, 128, 1), 0), out=buf624)
    del permute_845
    buf625 = reinterpret_tensor(buf597, (16, 128, 128), (16384, 128, 1), 0); del buf597  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf623, (16, 128, 128), (16384, 128, 1), 0), permute_846, out=buf625)
    del permute_846
    buf626 = reinterpret_tensor(buf623, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf623  # reuse
    cpp_fused_clone_142(c_void_p(tangents_15.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf626.data_ptr()))
    del tangents_15
    buf627 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf626, (2048, 128), (1, 2048), 0), view_134, out=buf627)
    buf628 = reinterpret_tensor(buf620, (128, 2048), (2048, 1), 0); del buf620  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf626, (128, 2048), (2048, 1), 0), permute_853, out=buf628)
    del permute_853
    buf629 = buf626; del buf626  # reuse
    cpp_fused_clone_143(c_void_p(tangents_14.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf629.data_ptr()))
    del tangents_14
    buf630 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (2048, 128), (1, 2048), 0), view_134, out=buf630)
    buf631 = reinterpret_tensor(buf624, (128, 2048), (2048, 1), 0); del buf624  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (128, 2048), (2048, 1), 0), permute_857, out=buf631)
    del permute_857
    buf632 = reinterpret_tensor(buf629, (128, 2048), (2048, 1), 0); del buf629  # reuse
    cpp_fused_view_144(c_void_p(buf625.data_ptr()), c_void_p(buf632.data_ptr()))
    buf633 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf632, (2048, 128), (1, 2048), 0), view_134, out=buf633)
    del view_134
    buf634 = reinterpret_tensor(buf625, (128, 2048), (2048, 1), 0); del buf625  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf632, permute_861, out=buf634)
    del permute_861
    buf635 = buf613; del buf613  # reuse
    buf636 = buf612; del buf612  # reuse
    buf637 = reinterpret_tensor(buf622, (2048, ), (1, ), 0); del buf622  # reuse
    buf638 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf639 = buf616; del buf616  # reuse
    cpp_fused_add_native_layer_norm_backward_145(c_void_p(buf639.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()))
    del div_60
    del mul_48
    del primals_81
    buf640 = reinterpret_tensor(buf608, (128, 8192), (8192, 1), 0); del buf608  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf639, (128, 2048), (2048, 1), 0), permute_863, out=buf640)
    del permute_863
    buf641 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf639, (2048, 128), (1, 2048), 0), view_132, out=buf641)
    del view_132
    buf642 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf643 = reinterpret_tensor(buf640, (1, 128, 8192), (1048576, 8192, 1), 0); del buf640  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_146(c_void_p(buf643.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf642.data_ptr()))
    del addmm_16
    del tanh_5
    buf644 = buf634; del buf634  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf643, (128, 8192), (8192, 1), 0), permute_867, out=buf644)
    del permute_867
    buf645 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf643, (8192, 128), (1, 8192), 0), view_130, out=buf645)
    del view_130
    buf646 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf647 = buf636; del buf636  # reuse
    buf648 = buf635; del buf635  # reuse
    buf649 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf650 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf651 = buf639; del buf639  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_147(c_void_p(buf651.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()))
    del div_61
    del mul_42
    del primals_75
    buf652 = buf644; del buf644  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf651, (128, 2048), (2048, 1), 0), permute_871, out=buf652)
    del permute_871
    buf653 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf651, (2048, 128), (1, 2048), 0), view_128, out=buf653)
    del view_128
    buf654 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_148(c_void_p(buf651.data_ptr()), c_void_p(buf654.data_ptr()))
    buf655 = reinterpret_tensor(buf631, (16, 128, 128), (16384, 128, 1), 0); del buf631  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_876, reinterpret_tensor(buf652, (16, 128, 128), (128, 2048, 1), 0), out=buf655)
    del permute_876
    buf656 = reinterpret_tensor(buf628, (16, 128, 128), (16384, 128, 1), 0); del buf628  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf652, (16, 128, 128), (128, 2048, 1), 0), permute_877, out=buf656)
    del permute_877
    buf657 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf658 = reinterpret_tensor(buf656, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf656  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_149(c_void_p(buf658.data_ptr()), c_void_p(alias_85.data_ptr()), c_void_p(slice_24.data_ptr()), c_void_p(buf657.data_ptr()))
    del alias_85
    del slice_24
    buf659 = reinterpret_tensor(buf652, (16, 128, 128), (16384, 128, 1), 0); del buf652  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_878, reinterpret_tensor(buf658, (16, 128, 128), (16384, 128, 1), 0), out=buf659)
    del permute_878
    buf660 = reinterpret_tensor(buf632, (16, 128, 128), (16384, 128, 1), 0); del buf632  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf658, (16, 128, 128), (16384, 128, 1), 0), permute_879, out=buf660)
    del permute_879
    buf661 = reinterpret_tensor(buf658, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf658  # reuse
    cpp_fused_clone_150(c_void_p(tangents_13.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf661.data_ptr()))
    del tangents_13
    buf662 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf661, (2048, 128), (1, 2048), 0), view_112, out=buf662)
    buf663 = reinterpret_tensor(buf655, (128, 2048), (2048, 1), 0); del buf655  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf661, (128, 2048), (2048, 1), 0), permute_886, out=buf663)
    del permute_886
    buf664 = buf661; del buf661  # reuse
    cpp_fused_clone_151(c_void_p(tangents_12.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf664.data_ptr()))
    del tangents_12
    buf665 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf664, (2048, 128), (1, 2048), 0), view_112, out=buf665)
    buf666 = reinterpret_tensor(buf659, (128, 2048), (2048, 1), 0); del buf659  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf664, (128, 2048), (2048, 1), 0), permute_890, out=buf666)
    del permute_890
    buf667 = reinterpret_tensor(buf664, (128, 2048), (2048, 1), 0); del buf664  # reuse
    cpp_fused_view_152(c_void_p(buf660.data_ptr()), c_void_p(buf667.data_ptr()))
    buf668 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf667, (2048, 128), (1, 2048), 0), view_112, out=buf668)
    del view_112
    buf669 = reinterpret_tensor(buf660, (128, 2048), (2048, 1), 0); del buf660  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf667, permute_894, out=buf669)
    del permute_894
    buf670 = buf648; del buf648  # reuse
    buf671 = buf647; del buf647  # reuse
    buf672 = reinterpret_tensor(buf657, (2048, ), (1, ), 0); del buf657  # reuse
    buf673 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf674 = buf651; del buf651  # reuse
    cpp_fused_add_native_layer_norm_backward_153(c_void_p(buf674.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_62.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()))
    del div_62
    del mul_40
    del primals_68
    buf675 = reinterpret_tensor(buf643, (128, 8192), (8192, 1), 0); del buf643  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf674, (128, 2048), (2048, 1), 0), permute_896, out=buf675)
    del permute_896
    buf676 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf674, (2048, 128), (1, 2048), 0), view_110, out=buf676)
    del view_110
    buf677 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf678 = reinterpret_tensor(buf675, (1, 128, 8192), (1048576, 8192, 1), 0); del buf675  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_154(c_void_p(buf678.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf677.data_ptr()))
    del addmm_13
    del tanh_4
    buf679 = buf669; del buf669  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf678, (128, 8192), (8192, 1), 0), permute_900, out=buf679)
    del permute_900
    buf680 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf678, (8192, 128), (1, 8192), 0), view_108, out=buf680)
    del view_108
    buf681 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf682 = buf671; del buf671  # reuse
    buf683 = buf670; del buf670  # reuse
    buf684 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf685 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf686 = buf674; del buf674  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_155(c_void_p(buf686.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf685.data_ptr()))
    del div_63
    del mul_34
    del primals_62
    buf687 = buf679; del buf679  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf686, (128, 2048), (2048, 1), 0), permute_904, out=buf687)
    del permute_904
    buf688 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf686, (2048, 128), (1, 2048), 0), view_106, out=buf688)
    del view_106
    buf689 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_156(c_void_p(buf686.data_ptr()), c_void_p(buf689.data_ptr()))
    buf690 = reinterpret_tensor(buf666, (16, 128, 128), (16384, 128, 1), 0); del buf666  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_909, reinterpret_tensor(buf687, (16, 128, 128), (128, 2048, 1), 0), out=buf690)
    del permute_909
    buf691 = reinterpret_tensor(buf663, (16, 128, 128), (16384, 128, 1), 0); del buf663  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf687, (16, 128, 128), (128, 2048, 1), 0), permute_910, out=buf691)
    del permute_910
    buf692 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf693 = reinterpret_tensor(buf691, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf691  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_157(c_void_p(buf693.data_ptr()), c_void_p(alias_87.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf692.data_ptr()))
    del alias_87
    del slice_20
    buf694 = reinterpret_tensor(buf687, (16, 128, 128), (16384, 128, 1), 0); del buf687  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_911, reinterpret_tensor(buf693, (16, 128, 128), (16384, 128, 1), 0), out=buf694)
    del permute_911
    buf695 = reinterpret_tensor(buf667, (16, 128, 128), (16384, 128, 1), 0); del buf667  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf693, (16, 128, 128), (16384, 128, 1), 0), permute_912, out=buf695)
    del permute_912
    buf696 = reinterpret_tensor(buf693, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf693  # reuse
    cpp_fused_clone_158(c_void_p(tangents_11.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf696.data_ptr()))
    del tangents_11
    buf697 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf696, (2048, 128), (1, 2048), 0), view_90, out=buf697)
    buf698 = reinterpret_tensor(buf690, (128, 2048), (2048, 1), 0); del buf690  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf696, (128, 2048), (2048, 1), 0), permute_919, out=buf698)
    del permute_919
    buf699 = buf696; del buf696  # reuse
    cpp_fused_clone_159(c_void_p(tangents_10.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf699.data_ptr()))
    del tangents_10
    buf700 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf699, (2048, 128), (1, 2048), 0), view_90, out=buf700)
    buf701 = reinterpret_tensor(buf694, (128, 2048), (2048, 1), 0); del buf694  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf699, (128, 2048), (2048, 1), 0), permute_923, out=buf701)
    del permute_923
    buf702 = reinterpret_tensor(buf699, (128, 2048), (2048, 1), 0); del buf699  # reuse
    cpp_fused_view_160(c_void_p(buf695.data_ptr()), c_void_p(buf702.data_ptr()))
    buf703 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf702, (2048, 128), (1, 2048), 0), view_90, out=buf703)
    del view_90
    buf704 = reinterpret_tensor(buf695, (128, 2048), (2048, 1), 0); del buf695  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf702, permute_927, out=buf704)
    del permute_927
    buf705 = buf683; del buf683  # reuse
    buf706 = buf682; del buf682  # reuse
    buf707 = reinterpret_tensor(buf692, (2048, ), (1, ), 0); del buf692  # reuse
    buf708 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf709 = buf686; del buf686  # reuse
    cpp_fused_add_native_layer_norm_backward_161(c_void_p(buf709.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()))
    del div_64
    del mul_32
    del primals_55
    buf710 = reinterpret_tensor(buf678, (128, 8192), (8192, 1), 0); del buf678  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf709, (128, 2048), (2048, 1), 0), permute_929, out=buf710)
    del permute_929
    buf711 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf709, (2048, 128), (1, 2048), 0), view_88, out=buf711)
    del view_88
    buf712 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf713 = reinterpret_tensor(buf710, (1, 128, 8192), (1048576, 8192, 1), 0); del buf710  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_162(c_void_p(buf713.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf712.data_ptr()))
    del addmm_10
    del tanh_3
    buf714 = buf704; del buf704  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf713, (128, 8192), (8192, 1), 0), permute_933, out=buf714)
    del permute_933
    buf715 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf713, (8192, 128), (1, 8192), 0), view_86, out=buf715)
    del view_86
    buf716 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf717 = buf706; del buf706  # reuse
    buf718 = buf705; del buf705  # reuse
    buf719 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf720 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf721 = buf709; del buf709  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_163(c_void_p(buf721.data_ptr()), c_void_p(buf713.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_65.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()))
    del div_65
    del mul_26
    del primals_49
    buf722 = buf714; del buf714  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf721, (128, 2048), (2048, 1), 0), permute_937, out=buf722)
    del permute_937
    buf723 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf721, (2048, 128), (1, 2048), 0), view_84, out=buf723)
    del view_84
    buf724 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_164(c_void_p(buf721.data_ptr()), c_void_p(buf724.data_ptr()))
    buf725 = reinterpret_tensor(buf701, (16, 128, 128), (16384, 128, 1), 0); del buf701  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_942, reinterpret_tensor(buf722, (16, 128, 128), (128, 2048, 1), 0), out=buf725)
    del permute_942
    buf726 = reinterpret_tensor(buf698, (16, 128, 128), (16384, 128, 1), 0); del buf698  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf722, (16, 128, 128), (128, 2048, 1), 0), permute_943, out=buf726)
    del permute_943
    buf727 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf728 = reinterpret_tensor(buf726, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf726  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_165(c_void_p(buf728.data_ptr()), c_void_p(alias_89.data_ptr()), c_void_p(slice_16.data_ptr()), c_void_p(buf727.data_ptr()))
    del alias_89
    del slice_16
    buf729 = reinterpret_tensor(buf722, (16, 128, 128), (16384, 128, 1), 0); del buf722  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_944, reinterpret_tensor(buf728, (16, 128, 128), (16384, 128, 1), 0), out=buf729)
    del permute_944
    buf730 = reinterpret_tensor(buf702, (16, 128, 128), (16384, 128, 1), 0); del buf702  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf728, (16, 128, 128), (16384, 128, 1), 0), permute_945, out=buf730)
    del permute_945
    buf731 = reinterpret_tensor(buf728, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf728  # reuse
    cpp_fused_clone_166(c_void_p(tangents_9.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf731.data_ptr()))
    del tangents_9
    buf732 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf731, (2048, 128), (1, 2048), 0), view_68, out=buf732)
    buf733 = reinterpret_tensor(buf725, (128, 2048), (2048, 1), 0); del buf725  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf731, (128, 2048), (2048, 1), 0), permute_952, out=buf733)
    del permute_952
    buf734 = buf731; del buf731  # reuse
    cpp_fused_clone_167(c_void_p(tangents_8.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf734.data_ptr()))
    del tangents_8
    buf735 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf734, (2048, 128), (1, 2048), 0), view_68, out=buf735)
    buf736 = reinterpret_tensor(buf729, (128, 2048), (2048, 1), 0); del buf729  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf734, (128, 2048), (2048, 1), 0), permute_956, out=buf736)
    del permute_956
    buf737 = reinterpret_tensor(buf734, (128, 2048), (2048, 1), 0); del buf734  # reuse
    cpp_fused_view_168(c_void_p(buf730.data_ptr()), c_void_p(buf737.data_ptr()))
    buf738 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf737, (2048, 128), (1, 2048), 0), view_68, out=buf738)
    del view_68
    buf739 = reinterpret_tensor(buf730, (128, 2048), (2048, 1), 0); del buf730  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf737, permute_960, out=buf739)
    del permute_960
    buf740 = buf718; del buf718  # reuse
    buf741 = buf717; del buf717  # reuse
    buf742 = reinterpret_tensor(buf727, (2048, ), (1, ), 0); del buf727  # reuse
    buf743 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf744 = buf721; del buf721  # reuse
    cpp_fused_add_native_layer_norm_backward_169(c_void_p(buf744.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf743.data_ptr()))
    del div_66
    del mul_24
    del primals_42
    buf745 = reinterpret_tensor(buf713, (128, 8192), (8192, 1), 0); del buf713  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf744, (128, 2048), (2048, 1), 0), permute_962, out=buf745)
    del permute_962
    buf746 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf744, (2048, 128), (1, 2048), 0), view_66, out=buf746)
    del view_66
    buf747 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf748 = reinterpret_tensor(buf745, (1, 128, 8192), (1048576, 8192, 1), 0); del buf745  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_170(c_void_p(buf748.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf747.data_ptr()))
    del addmm_7
    del tanh_2
    buf749 = buf739; del buf739  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (128, 8192), (8192, 1), 0), permute_966, out=buf749)
    del permute_966
    buf750 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (8192, 128), (1, 8192), 0), view_64, out=buf750)
    del view_64
    buf751 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf752 = buf741; del buf741  # reuse
    buf753 = buf740; del buf740  # reuse
    buf754 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf755 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf756 = buf744; del buf744  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_171(c_void_p(buf756.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_67.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf755.data_ptr()))
    del div_67
    del mul_18
    del primals_36
    buf757 = buf749; del buf749  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf756, (128, 2048), (2048, 1), 0), permute_970, out=buf757)
    del permute_970
    buf758 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf756, (2048, 128), (1, 2048), 0), view_62, out=buf758)
    del view_62
    buf759 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_172(c_void_p(buf756.data_ptr()), c_void_p(buf759.data_ptr()))
    buf760 = reinterpret_tensor(buf736, (16, 128, 128), (16384, 128, 1), 0); del buf736  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_975, reinterpret_tensor(buf757, (16, 128, 128), (128, 2048, 1), 0), out=buf760)
    del permute_975
    buf761 = reinterpret_tensor(buf733, (16, 128, 128), (16384, 128, 1), 0); del buf733  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf757, (16, 128, 128), (128, 2048, 1), 0), permute_976, out=buf761)
    del permute_976
    buf762 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf763 = reinterpret_tensor(buf761, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf761  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_173(c_void_p(buf763.data_ptr()), c_void_p(alias_91.data_ptr()), c_void_p(slice_12.data_ptr()), c_void_p(buf762.data_ptr()))
    del alias_91
    del slice_12
    buf764 = reinterpret_tensor(buf757, (16, 128, 128), (16384, 128, 1), 0); del buf757  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_977, reinterpret_tensor(buf763, (16, 128, 128), (16384, 128, 1), 0), out=buf764)
    del permute_977
    buf765 = reinterpret_tensor(buf737, (16, 128, 128), (16384, 128, 1), 0); del buf737  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf763, (16, 128, 128), (16384, 128, 1), 0), permute_978, out=buf765)
    del permute_978
    buf766 = reinterpret_tensor(buf763, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf763  # reuse
    cpp_fused_clone_174(c_void_p(tangents_7.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf766.data_ptr()))
    del tangents_7
    buf767 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf766, (2048, 128), (1, 2048), 0), view_46, out=buf767)
    buf768 = reinterpret_tensor(buf760, (128, 2048), (2048, 1), 0); del buf760  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf766, (128, 2048), (2048, 1), 0), permute_985, out=buf768)
    del permute_985
    buf769 = buf766; del buf766  # reuse
    cpp_fused_clone_175(c_void_p(tangents_6.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf769.data_ptr()))
    del tangents_6
    buf770 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf769, (2048, 128), (1, 2048), 0), view_46, out=buf770)
    buf771 = reinterpret_tensor(buf764, (128, 2048), (2048, 1), 0); del buf764  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf769, (128, 2048), (2048, 1), 0), permute_989, out=buf771)
    del permute_989
    buf772 = reinterpret_tensor(buf769, (128, 2048), (2048, 1), 0); del buf769  # reuse
    cpp_fused_view_176(c_void_p(buf765.data_ptr()), c_void_p(buf772.data_ptr()))
    buf773 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf772, (2048, 128), (1, 2048), 0), view_46, out=buf773)
    del view_46
    buf774 = reinterpret_tensor(buf765, (128, 2048), (2048, 1), 0); del buf765  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf772, permute_993, out=buf774)
    del permute_993
    buf775 = buf753; del buf753  # reuse
    buf776 = buf752; del buf752  # reuse
    buf777 = reinterpret_tensor(buf762, (2048, ), (1, ), 0); del buf762  # reuse
    buf778 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf779 = buf756; del buf756  # reuse
    cpp_fused_add_native_layer_norm_backward_177(c_void_p(buf779.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_68.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()))
    del div_68
    del mul_16
    del primals_29
    buf780 = reinterpret_tensor(buf748, (128, 8192), (8192, 1), 0); del buf748  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf779, (128, 2048), (2048, 1), 0), permute_995, out=buf780)
    del permute_995
    buf781 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf779, (2048, 128), (1, 2048), 0), view_44, out=buf781)
    del view_44
    buf782 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf783 = reinterpret_tensor(buf780, (1, 128, 8192), (1048576, 8192, 1), 0); del buf780  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_178(c_void_p(buf783.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf782.data_ptr()))
    del addmm_4
    del tanh_1
    buf784 = buf774; del buf774  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf783, (128, 8192), (8192, 1), 0), permute_999, out=buf784)
    del permute_999
    buf785 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf783, (8192, 128), (1, 8192), 0), view_42, out=buf785)
    del view_42
    buf786 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf787 = buf776; del buf776  # reuse
    buf788 = buf775; del buf775  # reuse
    buf789 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf790 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf791 = buf779; del buf779  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_179(c_void_p(buf791.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_69.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(buf789.data_ptr()), c_void_p(buf790.data_ptr()))
    del div_69
    del mul_10
    del primals_23
    buf792 = buf784; del buf784  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf791, (128, 2048), (2048, 1), 0), permute_1003, out=buf792)
    del permute_1003
    buf793 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf791, (2048, 128), (1, 2048), 0), view_40, out=buf793)
    del view_40
    buf794 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_180(c_void_p(buf791.data_ptr()), c_void_p(buf794.data_ptr()))
    buf795 = reinterpret_tensor(buf771, (16, 128, 128), (16384, 128, 1), 0); del buf771  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1008, reinterpret_tensor(buf792, (16, 128, 128), (128, 2048, 1), 0), out=buf795)
    del permute_1008
    buf796 = reinterpret_tensor(buf768, (16, 128, 128), (16384, 128, 1), 0); del buf768  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf792, (16, 128, 128), (128, 2048, 1), 0), permute_1009, out=buf796)
    del permute_1009
    buf797 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf798 = reinterpret_tensor(buf796, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf796  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_181(c_void_p(buf798.data_ptr()), c_void_p(alias_93.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf797.data_ptr()))
    del alias_93
    del slice_8
    buf799 = reinterpret_tensor(buf792, (16, 128, 128), (16384, 128, 1), 0); del buf792  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1010, reinterpret_tensor(buf798, (16, 128, 128), (16384, 128, 1), 0), out=buf799)
    del permute_1010
    buf800 = reinterpret_tensor(buf772, (16, 128, 128), (16384, 128, 1), 0); del buf772  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf798, (16, 128, 128), (16384, 128, 1), 0), permute_1011, out=buf800)
    del permute_1011
    buf801 = reinterpret_tensor(buf798, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf798  # reuse
    cpp_fused_clone_182(c_void_p(tangents_5.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf801.data_ptr()))
    del tangents_5
    buf802 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf801, (2048, 128), (1, 2048), 0), view_24, out=buf802)
    buf803 = reinterpret_tensor(buf795, (128, 2048), (2048, 1), 0); del buf795  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf801, (128, 2048), (2048, 1), 0), permute_1018, out=buf803)
    del permute_1018
    buf804 = buf801; del buf801  # reuse
    cpp_fused_clone_183(c_void_p(tangents_4.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf804.data_ptr()))
    del tangents_4
    buf805 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (2048, 128), (1, 2048), 0), view_24, out=buf805)
    buf806 = reinterpret_tensor(buf799, (128, 2048), (2048, 1), 0); del buf799  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (128, 2048), (2048, 1), 0), permute_1022, out=buf806)
    del permute_1022
    buf807 = reinterpret_tensor(buf804, (128, 2048), (2048, 1), 0); del buf804  # reuse
    cpp_fused_view_184(c_void_p(buf800.data_ptr()), c_void_p(buf807.data_ptr()))
    buf808 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf807, (2048, 128), (1, 2048), 0), view_24, out=buf808)
    del view_24
    buf809 = reinterpret_tensor(buf800, (128, 2048), (2048, 1), 0); del buf800  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf807, permute_1026, out=buf809)
    del permute_1026
    buf810 = buf788; del buf788  # reuse
    buf811 = buf787; del buf787  # reuse
    buf812 = reinterpret_tensor(buf797, (2048, ), (1, ), 0); del buf797  # reuse
    buf813 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf814 = buf791; del buf791  # reuse
    cpp_fused_add_native_layer_norm_backward_185(c_void_p(buf814.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_70.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()))
    del div_70
    del mul_8
    del primals_16
    buf815 = reinterpret_tensor(buf783, (128, 8192), (8192, 1), 0); del buf783  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf814, (128, 2048), (2048, 1), 0), permute_1028, out=buf815)
    del permute_1028
    buf816 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf814, (2048, 128), (1, 2048), 0), view_22, out=buf816)
    del view_22
    buf817 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf818 = reinterpret_tensor(buf815, (1, 128, 8192), (1048576, 8192, 1), 0); del buf815  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_186(c_void_p(buf818.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf817.data_ptr()))
    del addmm_1
    del tanh
    buf819 = buf809; del buf809  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf818, (128, 8192), (8192, 1), 0), permute_1032, out=buf819)
    del permute_1032
    buf820 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf818, (8192, 128), (1, 8192), 0), view_20, out=buf820)
    del view_20
    buf821 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf822 = buf811; del buf811  # reuse
    buf823 = buf810; del buf810  # reuse
    buf824 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf825 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf826 = buf814; del buf814  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_187(c_void_p(buf826.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_71.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()))
    del buf818
    del div_71
    del mul_2
    del primals_10
    buf827 = buf819; del buf819  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf826, (128, 2048), (2048, 1), 0), permute_1036, out=buf827)
    del permute_1036
    buf828 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf826, (2048, 128), (1, 2048), 0), view_18, out=buf828)
    del view_18
    buf829 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_188(c_void_p(buf826.data_ptr()), c_void_p(buf829.data_ptr()))
    buf830 = reinterpret_tensor(buf806, (16, 128, 128), (16384, 128, 1), 0); del buf806  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1041, reinterpret_tensor(buf827, (16, 128, 128), (128, 2048, 1), 0), out=buf830)
    del permute_1041
    buf831 = reinterpret_tensor(buf803, (16, 128, 128), (16384, 128, 1), 0); del buf803  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf827, (16, 128, 128), (128, 2048, 1), 0), permute_1042, out=buf831)
    del permute_1042
    buf832 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf833 = reinterpret_tensor(buf831, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf831  # reuse
    cpp_fused__softmax_backward_data_scalar_tensor_where_189(c_void_p(buf833.data_ptr()), c_void_p(alias_95.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(buf832.data_ptr()))
    del alias_95
    del slice_4
    buf834 = reinterpret_tensor(buf827, (16, 128, 128), (16384, 128, 1), 0); del buf827  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1043, reinterpret_tensor(buf833, (16, 128, 128), (16384, 128, 1), 0), out=buf834)
    del permute_1043
    buf835 = reinterpret_tensor(buf807, (16, 128, 128), (16384, 128, 1), 0); del buf807  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf833, (16, 128, 128), (16384, 128, 1), 0), permute_1044, out=buf835)
    del permute_1044
    buf836 = reinterpret_tensor(buf833, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf833  # reuse
    cpp_fused_clone_190(c_void_p(tangents_3.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(buf836.data_ptr()))
    del tangents_3
    buf837 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf836, (2048, 128), (1, 2048), 0), view_2, out=buf837)
    buf838 = reinterpret_tensor(buf830, (128, 2048), (2048, 1), 0); del buf830  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf836, (128, 2048), (2048, 1), 0), permute_1051, out=buf838)
    del permute_1051
    buf839 = buf836; del buf836  # reuse
    cpp_fused_clone_191(c_void_p(tangents_2.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf839.data_ptr()))
    del tangents_2
    buf840 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf839, (2048, 128), (1, 2048), 0), view_2, out=buf840)
    buf841 = reinterpret_tensor(buf834, (128, 2048), (2048, 1), 0); del buf834  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf839, (128, 2048), (2048, 1), 0), permute_1055, out=buf841)
    del permute_1055
    buf842 = reinterpret_tensor(buf839, (128, 2048), (2048, 1), 0); del buf839  # reuse
    cpp_fused_view_192(c_void_p(buf835.data_ptr()), c_void_p(buf842.data_ptr()))
    buf843 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf842, (2048, 128), (1, 2048), 0), view_2, out=buf843)
    del view_2
    buf844 = reinterpret_tensor(buf835, (128, 2048), (2048, 1), 0); del buf835  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf842, permute_1059, out=buf844)
    del permute_1059
    buf845 = buf823; del buf823  # reuse
    buf846 = buf822; del buf822  # reuse
    buf847 = reinterpret_tensor(buf832, (2048, ), (1, ), 0); del buf832  # reuse
    buf848 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf849 = buf826; del buf826  # reuse
    buf855 = reinterpret_tensor(buf842, (1, 128, 2048), (262144, 2048, 1), 0); del buf842  # reuse
    buf850 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    buf851 = buf849; del buf849  # reuse
    cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_193(c_void_p(buf851.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf841.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf845.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf850.data_ptr()))
    del buf838
    del buf841
    del buf844
    del buf845
    del buf846
    del div_72
    del mul
    del primals_3
    aten.index_put_(buf850, [view_1], buf851, True)
    del buf851
    del view_1
    buf854 = empty((50257, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_194(c_void_p(buf854.data_ptr()))
    aten.index_put_(buf854, [view], buf855, True)
    del buf855
    del view
    return (buf854, buf850, buf847, buf848, reinterpret_tensor(buf843, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf840, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf837, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf828, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf829, (2048, ), (1, ), 0), buf824, buf825, reinterpret_tensor(buf820, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf821, (8192, ), (1, ), 0), reinterpret_tensor(buf816, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf817, (2048, ), (1, ), 0), buf812, buf813, reinterpret_tensor(buf808, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf805, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf802, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf793, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf794, (2048, ), (1, ), 0), buf789, buf790, reinterpret_tensor(buf785, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf786, (8192, ), (1, ), 0), reinterpret_tensor(buf781, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf782, (2048, ), (1, ), 0), buf777, buf778, reinterpret_tensor(buf773, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf770, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf767, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf758, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf759, (2048, ), (1, ), 0), buf754, buf755, reinterpret_tensor(buf750, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf751, (8192, ), (1, ), 0), reinterpret_tensor(buf746, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf747, (2048, ), (1, ), 0), buf742, buf743, reinterpret_tensor(buf738, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf735, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf732, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf723, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf724, (2048, ), (1, ), 0), buf719, buf720, reinterpret_tensor(buf715, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf716, (8192, ), (1, ), 0), reinterpret_tensor(buf711, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf712, (2048, ), (1, ), 0), buf707, buf708, reinterpret_tensor(buf703, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf700, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf697, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf688, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf689, (2048, ), (1, ), 0), buf684, buf685, reinterpret_tensor(buf680, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf681, (8192, ), (1, ), 0), reinterpret_tensor(buf676, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf677, (2048, ), (1, ), 0), buf672, buf673, reinterpret_tensor(buf668, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf665, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf662, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf653, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf654, (2048, ), (1, ), 0), buf649, buf650, reinterpret_tensor(buf645, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf646, (8192, ), (1, ), 0), reinterpret_tensor(buf641, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf642, (2048, ), (1, ), 0), buf637, buf638, reinterpret_tensor(buf633, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf630, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf627, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf618, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf619, (2048, ), (1, ), 0), buf614, buf615, reinterpret_tensor(buf610, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf611, (8192, ), (1, ), 0), reinterpret_tensor(buf606, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf607, (2048, ), (1, ), 0), buf602, buf603, reinterpret_tensor(buf598, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf595, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf592, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf583, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf584, (2048, ), (1, ), 0), buf579, buf580, reinterpret_tensor(buf575, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf576, (8192, ), (1, ), 0), reinterpret_tensor(buf571, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf572, (2048, ), (1, ), 0), buf567, buf568, reinterpret_tensor(buf563, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf560, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf557, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf548, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf549, (2048, ), (1, ), 0), buf544, buf545, reinterpret_tensor(buf540, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf541, (8192, ), (1, ), 0), reinterpret_tensor(buf536, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf537, (2048, ), (1, ), 0), buf532, buf533, reinterpret_tensor(buf528, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf525, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf522, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf513, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf514, (2048, ), (1, ), 0), buf509, buf510, reinterpret_tensor(buf505, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf506, (8192, ), (1, ), 0), reinterpret_tensor(buf501, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf502, (2048, ), (1, ), 0), buf497, buf498, reinterpret_tensor(buf493, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf490, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf487, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf478, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf479, (2048, ), (1, ), 0), buf474, buf475, reinterpret_tensor(buf470, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf471, (8192, ), (1, ), 0), reinterpret_tensor(buf466, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf467, (2048, ), (1, ), 0), buf462, buf463, reinterpret_tensor(buf458, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf455, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf452, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf443, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf444, (2048, ), (1, ), 0), buf439, buf440, reinterpret_tensor(buf435, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf436, (8192, ), (1, ), 0), reinterpret_tensor(buf431, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf432, (2048, ), (1, ), 0), buf427, buf428, reinterpret_tensor(buf423, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf420, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf417, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf408, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf409, (2048, ), (1, ), 0), buf404, buf405, reinterpret_tensor(buf400, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf401, (8192, ), (1, ), 0), reinterpret_tensor(buf396, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf397, (2048, ), (1, ), 0), buf392, buf393, reinterpret_tensor(buf388, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf385, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf382, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf373, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf374, (2048, ), (1, ), 0), buf369, buf370, reinterpret_tensor(buf365, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf366, (8192, ), (1, ), 0), reinterpret_tensor(buf361, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf362, (2048, ), (1, ), 0), buf357, buf358, reinterpret_tensor(buf353, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf350, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf347, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf338, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf339, (2048, ), (1, ), 0), buf334, buf335, reinterpret_tensor(buf330, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf331, (8192, ), (1, ), 0), reinterpret_tensor(buf326, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf327, (2048, ), (1, ), 0), buf322, buf323, reinterpret_tensor(buf318, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf315, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf312, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf303, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf304, (2048, ), (1, ), 0), buf299, buf300, reinterpret_tensor(buf295, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf296, (8192, ), (1, ), 0), reinterpret_tensor(buf291, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf292, (2048, ), (1, ), 0), buf287, buf288, reinterpret_tensor(buf283, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf280, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf277, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf268, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf269, (2048, ), (1, ), 0), buf264, buf265, reinterpret_tensor(buf260, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf261, (8192, ), (1, ), 0), reinterpret_tensor(buf256, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf257, (2048, ), (1, ), 0), buf252, buf253, reinterpret_tensor(buf248, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf245, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf242, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf233, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf234, (2048, ), (1, ), 0), buf229, buf230, reinterpret_tensor(buf225, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf226, (8192, ), (1, ), 0), reinterpret_tensor(buf221, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf222, (2048, ), (1, ), 0), buf217, buf218, reinterpret_tensor(buf213, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf210, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf207, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf198, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf199, (2048, ), (1, ), 0), buf194, buf195, reinterpret_tensor(buf190, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf191, (8192, ), (1, ), 0), reinterpret_tensor(buf186, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf187, (2048, ), (1, ), 0), buf182, buf183, reinterpret_tensor(buf178, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf175, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf172, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf163, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf164, (2048, ), (1, ), 0), buf159, buf160, reinterpret_tensor(buf155, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf156, (8192, ), (1, ), 0), reinterpret_tensor(buf151, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf152, (2048, ), (1, ), 0), buf147, buf148, reinterpret_tensor(buf143, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf140, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf137, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf128, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf129, (2048, ), (1, ), 0), buf124, buf125, reinterpret_tensor(buf120, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf121, (8192, ), (1, ), 0), reinterpret_tensor(buf116, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf117, (2048, ), (1, ), 0), buf112, buf113, reinterpret_tensor(buf108, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf105, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf102, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf93, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf94, (2048, ), (1, ), 0), buf89, buf90, reinterpret_tensor(buf85, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf86, (8192, ), (1, ), 0), reinterpret_tensor(buf81, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf82, (2048, ), (1, ), 0), buf77, buf78, reinterpret_tensor(buf73, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf70, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf67, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf58, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf59, (2048, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf50, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf51, (8192, ), (1, ), 0), reinterpret_tensor(buf46, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf47, (2048, ), (1, ), 0), buf42, buf43, reinterpret_tensor(buf38, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf35, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf32, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf23, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf24, (2048, ), (1, ), 0), buf19, buf20, reinterpret_tensor(buf15, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf16, (8192, ), (1, ), 0), reinterpret_tensor(buf11, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf12, (2048, ), (1, ), 0), buf8, buf9, reinterpret_tensor(buf3, (2, 2048), (2048, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    view_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_18 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_40 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_10 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_62 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_18 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_24 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_84 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_26 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_106 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_34 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_128 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_28 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_150 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_156 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_32 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_172 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_58 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_36 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_194 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_40 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_216 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_74 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_222 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_44 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_238 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_244 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_48 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_260 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_52 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_282 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_98 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_284 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_37 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_12 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_286 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_288 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_56 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_304 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_106 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_306 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_13 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_308 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_310 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_60 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_326 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_114 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_328 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_43 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_14 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_330 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_332 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_64 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_348 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_122 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_350 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_15 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_352 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_354 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_68 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_370 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_130 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_372 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_49 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_16 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_374 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_136 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_376 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_72 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_392 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_394 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_17 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_396 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_144 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_398 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_76 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_414 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_146 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_416 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_55 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_18 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_418 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_420 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_80 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_436 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_154 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_438 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_19 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_440 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_160 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_442 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_84 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_458 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_162 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_460 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_61 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_20 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_462 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_168 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_464 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_88 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_480 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_170 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_482 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_21 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_484 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_486 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_92 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_502 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_178 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_504 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_67 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_22 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_506 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_184 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_508 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_96 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_524 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_186 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_526 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_23 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_528 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_531 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    sub_73 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    full_default_24 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    permute_267 = rand_strided((2, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_49 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_51 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_329 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_333 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_339 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_53 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_351 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_362 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_55 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_384 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_391 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_395 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_399 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_401 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_57 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_416 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_417 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_424 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_428 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_432 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_442 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_59 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_449 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_457 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_461 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_465 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_61 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_483 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_490 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_494 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_498 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_504 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_508 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_513 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_63 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_515 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_516 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_523 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_527 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_531 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_533 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_537 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_541 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_546 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_547 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_65 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_548 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_549 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_556 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_560 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_564 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_566 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_570 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_574 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_579 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_580 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_67 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_581 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_582 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_589 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_593 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_597 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_599 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_603 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_607 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_612 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_613 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_69 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_614 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_615 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_622 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_626 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_630 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_632 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_636 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_640 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_645 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_646 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_71 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_647 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_648 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_655 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_659 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_663 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_665 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_669 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_673 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_678 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_679 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_73 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_680 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_681 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_688 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_692 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_696 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_698 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_702 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_706 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_711 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_712 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_75 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_713 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_714 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_721 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_725 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_729 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_731 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_735 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_53 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_739 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_744 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_745 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_77 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_746 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_747 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_754 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_758 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_762 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_764 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_768 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_772 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_777 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_778 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_79 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_779 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_780 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_787 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_791 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_795 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_797 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_801 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_805 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_810 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_811 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_81 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_812 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_813 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_820 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_824 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_828 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_830 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_834 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_59 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_838 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_843 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_844 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_83 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_845 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_846 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_853 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_857 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_861 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_863 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_867 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_871 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_876 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_877 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_85 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_878 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_879 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_886 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_890 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_894 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_62 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_896 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_900 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_904 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_909 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_910 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_87 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_911 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_912 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_919 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_923 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_927 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_929 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_933 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_65 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_937 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_942 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_943 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_89 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_944 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_945 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_952 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_956 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_960 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_962 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_966 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_970 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_975 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_976 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_91 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_977 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_978 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_985 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_989 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_993 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_68 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_995 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_999 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_69 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1003 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1008 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1009 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_93 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1010 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_1011 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_1018 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1022 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1026 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_70 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1028 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_1032 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_71 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1036 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1041 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1042 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_95 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1043 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_1044 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_1051 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1055 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1059 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_27 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_28 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_29 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_30 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_31 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_32 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_33 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_34 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_35 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_36 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_37 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_38 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_39 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_40 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_41 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_42 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_43 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_44 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_45 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_46 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_47 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_48 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_49 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_50 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, view, view_1, mul, view_2, slice_4, view_18, mul_2, view_20, addmm_1, tanh, view_22, mul_8, view_24, slice_8, view_40, mul_10, view_42, addmm_4, tanh_1, view_44, mul_16, view_46, slice_12, view_62, mul_18, view_64, addmm_7, tanh_2, view_66, mul_24, view_68, slice_16, view_84, mul_26, view_86, addmm_10, tanh_3, view_88, mul_32, view_90, slice_20, view_106, mul_34, view_108, addmm_13, tanh_4, view_110, mul_40, view_112, slice_24, view_128, mul_42, view_130, addmm_16, tanh_5, view_132, mul_48, view_134, slice_28, view_150, mul_50, view_152, addmm_19, tanh_6, view_154, mul_56, view_156, slice_32, view_172, mul_58, view_174, addmm_22, tanh_7, view_176, mul_64, view_178, slice_36, view_194, mul_66, view_196, addmm_25, tanh_8, view_198, mul_72, view_200, slice_40, view_216, mul_74, view_218, addmm_28, tanh_9, view_220, mul_80, view_222, slice_44, view_238, mul_82, view_240, addmm_31, tanh_10, view_242, mul_88, view_244, slice_48, view_260, mul_90, view_262, addmm_34, tanh_11, view_264, mul_96, view_266, slice_52, view_282, mul_98, view_284, addmm_37, tanh_12, view_286, mul_104, view_288, slice_56, view_304, mul_106, view_306, addmm_40, tanh_13, view_308, mul_112, view_310, slice_60, view_326, mul_114, view_328, addmm_43, tanh_14, view_330, mul_120, view_332, slice_64, view_348, mul_122, view_350, addmm_46, tanh_15, view_352, mul_128, view_354, slice_68, view_370, mul_130, view_372, addmm_49, tanh_16, view_374, mul_136, view_376, slice_72, view_392, mul_138, view_394, addmm_52, tanh_17, view_396, mul_144, view_398, slice_76, view_414, mul_146, view_416, addmm_55, tanh_18, view_418, mul_152, view_420, slice_80, view_436, mul_154, view_438, addmm_58, tanh_19, view_440, mul_160, view_442, slice_84, view_458, mul_162, view_460, addmm_61, tanh_20, view_462, mul_168, view_464, slice_88, view_480, mul_170, view_482, addmm_64, tanh_21, view_484, mul_176, view_486, slice_92, view_502, mul_178, view_504, addmm_67, tanh_22, view_506, mul_184, view_508, slice_96, view_524, mul_186, view_526, addmm_70, tanh_23, view_528, mul_192, view_531, sub_73, full_default_24, permute_267, div_24, permute_269, permute_273, div_25, permute_277, permute_282, permute_283, alias_49, permute_284, permute_285, permute_292, permute_296, permute_300, div_26, permute_302, permute_306, div_27, permute_310, permute_315, permute_316, alias_51, permute_317, permute_318, permute_325, permute_329, permute_333, div_28, permute_335, permute_339, div_29, permute_343, permute_348, permute_349, alias_53, permute_350, permute_351, permute_358, permute_362, permute_366, div_30, permute_368, permute_372, div_31, permute_376, permute_381, permute_382, alias_55, permute_383, permute_384, permute_391, permute_395, permute_399, div_32, permute_401, permute_405, div_33, permute_409, permute_414, permute_415, alias_57, permute_416, permute_417, permute_424, permute_428, permute_432, div_34, permute_434, permute_438, div_35, permute_442, permute_447, permute_448, alias_59, permute_449, permute_450, permute_457, permute_461, permute_465, div_36, permute_467, permute_471, div_37, permute_475, permute_480, permute_481, alias_61, permute_482, permute_483, permute_490, permute_494, permute_498, div_38, permute_500, permute_504, div_39, permute_508, permute_513, permute_514, alias_63, permute_515, permute_516, permute_523, permute_527, permute_531, div_40, permute_533, permute_537, div_41, permute_541, permute_546, permute_547, alias_65, permute_548, permute_549, permute_556, permute_560, permute_564, div_42, permute_566, permute_570, div_43, permute_574, permute_579, permute_580, alias_67, permute_581, permute_582, permute_589, permute_593, permute_597, div_44, permute_599, permute_603, div_45, permute_607, permute_612, permute_613, alias_69, permute_614, permute_615, permute_622, permute_626, permute_630, div_46, permute_632, permute_636, div_47, permute_640, permute_645, permute_646, alias_71, permute_647, permute_648, permute_655, permute_659, permute_663, div_48, permute_665, permute_669, div_49, permute_673, permute_678, permute_679, alias_73, permute_680, permute_681, permute_688, permute_692, permute_696, div_50, permute_698, permute_702, div_51, permute_706, permute_711, permute_712, alias_75, permute_713, permute_714, permute_721, permute_725, permute_729, div_52, permute_731, permute_735, div_53, permute_739, permute_744, permute_745, alias_77, permute_746, permute_747, permute_754, permute_758, permute_762, div_54, permute_764, permute_768, div_55, permute_772, permute_777, permute_778, alias_79, permute_779, permute_780, permute_787, permute_791, permute_795, div_56, permute_797, permute_801, div_57, permute_805, permute_810, permute_811, alias_81, permute_812, permute_813, permute_820, permute_824, permute_828, div_58, permute_830, permute_834, div_59, permute_838, permute_843, permute_844, alias_83, permute_845, permute_846, permute_853, permute_857, permute_861, div_60, permute_863, permute_867, div_61, permute_871, permute_876, permute_877, alias_85, permute_878, permute_879, permute_886, permute_890, permute_894, div_62, permute_896, permute_900, div_63, permute_904, permute_909, permute_910, alias_87, permute_911, permute_912, permute_919, permute_923, permute_927, div_64, permute_929, permute_933, div_65, permute_937, permute_942, permute_943, alias_89, permute_944, permute_945, permute_952, permute_956, permute_960, div_66, permute_962, permute_966, div_67, permute_970, permute_975, permute_976, alias_91, permute_977, permute_978, permute_985, permute_989, permute_993, div_68, permute_995, permute_999, div_69, permute_1003, permute_1008, permute_1009, alias_93, permute_1010, permute_1011, permute_1018, permute_1022, permute_1026, div_70, permute_1028, permute_1032, div_71, permute_1036, permute_1041, permute_1042, alias_95, permute_1043, permute_1044, permute_1051, permute_1055, permute_1059, div_72, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTNeoForSequenceClassification', benchmark_compiled_module)
