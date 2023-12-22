
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


cpp_fused_div_native_layer_norm_backward_sum_0 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1000L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp7 = tmp5 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp14 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = static_cast<float>(196.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp12 - tmp16;
                        auto tmp18 = at::vec::Vectorized<float>(tmp0);
                        auto tmp19 = tmp18 * tmp17;
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp3;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_2 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_6 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_10 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_14 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_18 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_22 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_26 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_30 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_34 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_38 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_42 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_46 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_sum_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.7071067811865476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp6.erf();
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 * tmp3;
                    auto tmp15 = static_cast<float>(-0.5);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp17.exp();
                    auto tmp19 = static_cast<float>(0.3989422804014327);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp3 * tmp21;
                    auto tmp23 = tmp13 + tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6144L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(768.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, mul, view_1, mm, view_3, mul_5, view_5, addmm_1, view_7, mul_10, view_9, mm_1, view_11, mul_15, view_13, addmm_4, view_15, mul_20, view_17, mm_2, view_19, mul_25, view_21, addmm_7, view_23, mul_30, view_25, mm_3, view_27, mul_35, view_29, addmm_10, view_31, mul_40, view_33, mm_4, view_35, mul_45, view_37, addmm_13, view_39, mul_50, view_41, mm_5, view_43, mul_55, view_45, addmm_16, view_47, mul_60, view_49, mm_6, view_51, mul_65, view_53, addmm_19, view_55, mul_70, view_57, mm_7, view_59, mul_75, view_61, addmm_22, view_63, mul_80, view_65, mm_8, view_67, mul_85, view_69, addmm_25, view_71, mul_90, view_73, mm_9, view_75, mul_95, view_77, addmm_28, view_79, mul_100, view_81, mm_10, view_83, mul_105, view_85, addmm_31, view_87, mul_110, view_89, mm_11, view_91, mul_115, view_93, addmm_34, view_95, mul_120, clone_85, permute_74, div_1, permute_78, permute_82, div_2, permute_87, permute_93, div_3, permute_96, permute_100, div_4, permute_105, permute_111, div_5, permute_114, permute_118, div_6, permute_123, permute_129, div_7, permute_132, permute_136, div_8, permute_141, permute_147, div_9, permute_150, permute_154, div_10, permute_159, permute_165, div_11, permute_168, permute_172, div_12, permute_177, permute_183, div_13, permute_186, permute_190, div_14, permute_195, permute_201, div_15, permute_204, permute_208, div_16, permute_213, permute_219, div_17, permute_222, permute_226, div_18, permute_231, permute_237, div_19, permute_240, permute_244, div_20, permute_249, permute_255, div_21, permute_258, permute_262, div_22, permute_267, permute_273, div_23, permute_276, permute_280, div_24, permute_285, permute_291, div_25, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_1, (6144, 196), (196, 1))
    assert_size_stride(mm, (6144, 384), (384, 1))
    assert_size_stride(view_3, (6144, 384), (384, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(addmm_1, (1568, 3072), (3072, 1))
    assert_size_stride(view_7, (1568, 3072), (3072, 1))
    assert_size_stride(mul_10, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_9, (6144, 196), (196, 1))
    assert_size_stride(mm_1, (6144, 384), (384, 1))
    assert_size_stride(view_11, (6144, 384), (384, 1))
    assert_size_stride(mul_15, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_13, (1568, 768), (768, 1))
    assert_size_stride(addmm_4, (1568, 3072), (3072, 1))
    assert_size_stride(view_15, (1568, 3072), (3072, 1))
    assert_size_stride(mul_20, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_17, (6144, 196), (196, 1))
    assert_size_stride(mm_2, (6144, 384), (384, 1))
    assert_size_stride(view_19, (6144, 384), (384, 1))
    assert_size_stride(mul_25, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_21, (1568, 768), (768, 1))
    assert_size_stride(addmm_7, (1568, 3072), (3072, 1))
    assert_size_stride(view_23, (1568, 3072), (3072, 1))
    assert_size_stride(mul_30, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_25, (6144, 196), (196, 1))
    assert_size_stride(mm_3, (6144, 384), (384, 1))
    assert_size_stride(view_27, (6144, 384), (384, 1))
    assert_size_stride(mul_35, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_29, (1568, 768), (768, 1))
    assert_size_stride(addmm_10, (1568, 3072), (3072, 1))
    assert_size_stride(view_31, (1568, 3072), (3072, 1))
    assert_size_stride(mul_40, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_33, (6144, 196), (196, 1))
    assert_size_stride(mm_4, (6144, 384), (384, 1))
    assert_size_stride(view_35, (6144, 384), (384, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_37, (1568, 768), (768, 1))
    assert_size_stride(addmm_13, (1568, 3072), (3072, 1))
    assert_size_stride(view_39, (1568, 3072), (3072, 1))
    assert_size_stride(mul_50, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_41, (6144, 196), (196, 1))
    assert_size_stride(mm_5, (6144, 384), (384, 1))
    assert_size_stride(view_43, (6144, 384), (384, 1))
    assert_size_stride(mul_55, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_45, (1568, 768), (768, 1))
    assert_size_stride(addmm_16, (1568, 3072), (3072, 1))
    assert_size_stride(view_47, (1568, 3072), (3072, 1))
    assert_size_stride(mul_60, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_49, (6144, 196), (196, 1))
    assert_size_stride(mm_6, (6144, 384), (384, 1))
    assert_size_stride(view_51, (6144, 384), (384, 1))
    assert_size_stride(mul_65, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_53, (1568, 768), (768, 1))
    assert_size_stride(addmm_19, (1568, 3072), (3072, 1))
    assert_size_stride(view_55, (1568, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (6144, 196), (196, 1))
    assert_size_stride(mm_7, (6144, 384), (384, 1))
    assert_size_stride(view_59, (6144, 384), (384, 1))
    assert_size_stride(mul_75, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_61, (1568, 768), (768, 1))
    assert_size_stride(addmm_22, (1568, 3072), (3072, 1))
    assert_size_stride(view_63, (1568, 3072), (3072, 1))
    assert_size_stride(mul_80, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_65, (6144, 196), (196, 1))
    assert_size_stride(mm_8, (6144, 384), (384, 1))
    assert_size_stride(view_67, (6144, 384), (384, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_69, (1568, 768), (768, 1))
    assert_size_stride(addmm_25, (1568, 3072), (3072, 1))
    assert_size_stride(view_71, (1568, 3072), (3072, 1))
    assert_size_stride(mul_90, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_73, (6144, 196), (196, 1))
    assert_size_stride(mm_9, (6144, 384), (384, 1))
    assert_size_stride(view_75, (6144, 384), (384, 1))
    assert_size_stride(mul_95, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_77, (1568, 768), (768, 1))
    assert_size_stride(addmm_28, (1568, 3072), (3072, 1))
    assert_size_stride(view_79, (1568, 3072), (3072, 1))
    assert_size_stride(mul_100, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_81, (6144, 196), (196, 1))
    assert_size_stride(mm_10, (6144, 384), (384, 1))
    assert_size_stride(view_83, (6144, 384), (384, 1))
    assert_size_stride(mul_105, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_85, (1568, 768), (768, 1))
    assert_size_stride(addmm_31, (1568, 3072), (3072, 1))
    assert_size_stride(view_87, (1568, 3072), (3072, 1))
    assert_size_stride(mul_110, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_89, (6144, 196), (196, 1))
    assert_size_stride(mm_11, (6144, 384), (384, 1))
    assert_size_stride(view_91, (6144, 384), (384, 1))
    assert_size_stride(mul_115, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_93, (1568, 768), (768, 1))
    assert_size_stride(addmm_34, (1568, 3072), (3072, 1))
    assert_size_stride(view_95, (1568, 3072), (3072, 1))
    assert_size_stride(mul_120, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(clone_85, (8, 768), (768, 1))
    assert_size_stride(permute_74, (1000, 768), (768, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_78, (768, 3072), (3072, 1))
    assert_size_stride(permute_82, (3072, 768), (768, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_87, (196, 384), (384, 1))
    assert_size_stride(permute_93, (384, 196), (196, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_96, (768, 3072), (3072, 1))
    assert_size_stride(permute_100, (3072, 768), (768, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_105, (196, 384), (384, 1))
    assert_size_stride(permute_111, (384, 196), (196, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_114, (768, 3072), (3072, 1))
    assert_size_stride(permute_118, (3072, 768), (768, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_123, (196, 384), (384, 1))
    assert_size_stride(permute_129, (384, 196), (196, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_141, (196, 384), (384, 1))
    assert_size_stride(permute_147, (384, 196), (196, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_159, (196, 384), (384, 1))
    assert_size_stride(permute_165, (384, 196), (196, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_168, (768, 3072), (3072, 1))
    assert_size_stride(permute_172, (3072, 768), (768, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_177, (196, 384), (384, 1))
    assert_size_stride(permute_183, (384, 196), (196, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_186, (768, 3072), (3072, 1))
    assert_size_stride(permute_190, (3072, 768), (768, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_195, (196, 384), (384, 1))
    assert_size_stride(permute_201, (384, 196), (196, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_204, (768, 3072), (3072, 1))
    assert_size_stride(permute_208, (3072, 768), (768, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_213, (196, 384), (384, 1))
    assert_size_stride(permute_219, (384, 196), (196, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_222, (768, 3072), (3072, 1))
    assert_size_stride(permute_226, (3072, 768), (768, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_231, (196, 384), (384, 1))
    assert_size_stride(permute_237, (384, 196), (196, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_249, (196, 384), (384, 1))
    assert_size_stride(permute_255, (384, 196), (196, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_258, (768, 3072), (3072, 1))
    assert_size_stride(permute_262, (3072, 768), (768, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_267, (196, 384), (384, 1))
    assert_size_stride(permute_273, (384, 196), (196, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_276, (768, 3072), (3072, 1))
    assert_size_stride(permute_280, (3072, 768), (768, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_285, (196, 384), (384, 1))
    assert_size_stride(permute_291, (384, 196), (196, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_74, out=buf0)
    del permute_74
    buf1 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_85, out=buf1)
    del clone_85
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf7 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_div_native_layer_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del div_1
    del mul_120
    del primals_147
    del tangents_1
    buf8 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1568, 768), (768, 1), 0), permute_78, out=buf8)
    del permute_78
    buf9 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (768, 1568), (1, 768), 0), view_95, out=buf9)
    del view_95
    buf10 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf8, (8, 196, 3072), (602112, 3072, 1), 0); del buf8  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf10.data_ptr()))
    del addmm_34
    buf12 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1568, 3072), (3072, 1), 0), permute_82, out=buf12)
    del permute_82
    buf13 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (3072, 1568), (1, 3072), 0), view_93, out=buf13)
    del view_93
    buf14 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 196, 768), (150528, 768, 1), 0); del buf12  # reuse
    buf20 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(mul_115.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del div_2
    del mul_115
    del primals_141
    buf21 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, permute_87, out=buf21)
    del permute_87
    buf22 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (196, 6144), (1, 196), 0), view_91, out=buf22)
    del view_91
    buf23 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf24 = reinterpret_tensor(buf21, (8, 768, 384), (294912, 384, 1), 0); del buf21  # reuse
    buf25 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_3(c_void_p(buf24.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()))
    del mm_11
    del primals_138
    buf26 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (384, 6144), (1, 384), 0), view_89, out=buf26)
    del view_89
    buf27 = buf20; del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (6144, 384), (384, 1), 0), permute_93, out=buf27)
    del permute_93
    buf28 = buf16; del buf16  # reuse
    buf29 = buf15; del buf15  # reuse
    buf30 = empty((768, ), device='cpu', dtype=torch.float32)
    buf31 = empty((768, ), device='cpu', dtype=torch.float32)
    buf32 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_backward_4(c_void_p(buf32.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(mul_110.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del div_3
    del mul_110
    del primals_135
    buf33 = reinterpret_tensor(buf11, (1568, 3072), (3072, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (1568, 768), (768, 1), 0), permute_96, out=buf33)
    del permute_96
    buf34 = reinterpret_tensor(buf24, (768, 3072), (3072, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (768, 1568), (1, 768), 0), view_87, out=buf34)
    del view_87
    buf35 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf33, (8, 196, 3072), (602112, 3072, 1), 0); del buf33  # reuse
    cpp_fused_gelu_gelu_backward_sum_5(c_void_p(buf36.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf35.data_ptr()))
    del addmm_31
    buf37 = reinterpret_tensor(buf27, (1568, 768), (768, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1568, 3072), (3072, 1), 0), permute_100, out=buf37)
    del permute_100
    buf38 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (3072, 1568), (1, 3072), 0), view_85, out=buf38)
    del view_85
    buf39 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf40 = buf29; del buf29  # reuse
    buf41 = buf28; del buf28  # reuse
    buf42 = empty((768, ), device='cpu', dtype=torch.float32)
    buf43 = empty((768, ), device='cpu', dtype=torch.float32)
    buf44 = buf32; del buf32  # reuse
    buf45 = reinterpret_tensor(buf5, (6144, 196), (196, 1), 0); del buf5  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_6(c_void_p(buf44.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(mul_105.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()))
    del div_4
    del mul_105
    del primals_129
    buf46 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_105, out=buf46)
    del permute_105
    buf47 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (196, 6144), (1, 196), 0), view_83, out=buf47)
    del view_83
    buf48 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf46, (8, 768, 384), (294912, 384, 1), 0); del buf46  # reuse
    buf50 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_7(c_void_p(buf49.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(mm_10.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del mm_10
    del primals_126
    buf51 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (384, 6144), (1, 384), 0), view_81, out=buf51)
    del view_81
    buf52 = buf45; del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (6144, 384), (384, 1), 0), permute_111, out=buf52)
    del permute_111
    buf53 = buf41; del buf41  # reuse
    buf54 = buf40; del buf40  # reuse
    buf55 = empty((768, ), device='cpu', dtype=torch.float32)
    buf56 = empty((768, ), device='cpu', dtype=torch.float32)
    buf57 = buf44; del buf44  # reuse
    cpp_fused_add_native_layer_norm_backward_8(c_void_p(buf57.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_100.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del div_5
    del mul_100
    del primals_123
    buf58 = reinterpret_tensor(buf36, (1568, 3072), (3072, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (1568, 768), (768, 1), 0), permute_114, out=buf58)
    del permute_114
    buf59 = reinterpret_tensor(buf49, (768, 3072), (3072, 1), 0); del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (768, 1568), (1, 768), 0), view_79, out=buf59)
    del view_79
    buf60 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf61 = reinterpret_tensor(buf58, (8, 196, 3072), (602112, 3072, 1), 0); del buf58  # reuse
    cpp_fused_gelu_gelu_backward_sum_9(c_void_p(buf61.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf60.data_ptr()))
    del addmm_28
    buf62 = reinterpret_tensor(buf52, (1568, 768), (768, 1), 0); del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0), permute_118, out=buf62)
    del permute_118
    buf63 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (3072, 1568), (1, 3072), 0), view_77, out=buf63)
    del view_77
    buf64 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf65 = buf54; del buf54  # reuse
    buf66 = buf53; del buf53  # reuse
    buf67 = empty((768, ), device='cpu', dtype=torch.float32)
    buf68 = empty((768, ), device='cpu', dtype=torch.float32)
    buf69 = buf57; del buf57  # reuse
    buf70 = reinterpret_tensor(buf37, (6144, 196), (196, 1), 0); del buf37  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_10(c_void_p(buf69.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_95.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    del div_6
    del mul_95
    del primals_117
    buf71 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, permute_123, out=buf71)
    del permute_123
    buf72 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (196, 6144), (1, 196), 0), view_75, out=buf72)
    del view_75
    buf73 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf71, (8, 768, 384), (294912, 384, 1), 0); del buf71  # reuse
    buf75 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_11(c_void_p(buf74.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    del mm_9
    del primals_114
    buf76 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (384, 6144), (1, 384), 0), view_73, out=buf76)
    del view_73
    buf77 = buf70; del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (6144, 384), (384, 1), 0), permute_129, out=buf77)
    del permute_129
    buf78 = buf66; del buf66  # reuse
    buf79 = buf65; del buf65  # reuse
    buf80 = empty((768, ), device='cpu', dtype=torch.float32)
    buf81 = empty((768, ), device='cpu', dtype=torch.float32)
    buf82 = buf69; del buf69  # reuse
    cpp_fused_add_native_layer_norm_backward_12(c_void_p(buf82.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del div_7
    del mul_90
    del primals_111
    buf83 = reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0); del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (1568, 768), (768, 1), 0), permute_132, out=buf83)
    del permute_132
    buf84 = reinterpret_tensor(buf74, (768, 3072), (3072, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (768, 1568), (1, 768), 0), view_71, out=buf84)
    del view_71
    buf85 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf83, (8, 196, 3072), (602112, 3072, 1), 0); del buf83  # reuse
    cpp_fused_gelu_gelu_backward_sum_13(c_void_p(buf86.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf85.data_ptr()))
    del addmm_25
    buf87 = reinterpret_tensor(buf77, (1568, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (1568, 3072), (3072, 1), 0), permute_136, out=buf87)
    del permute_136
    buf88 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (3072, 1568), (1, 3072), 0), view_69, out=buf88)
    del view_69
    buf89 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf90 = buf79; del buf79  # reuse
    buf91 = buf78; del buf78  # reuse
    buf92 = empty((768, ), device='cpu', dtype=torch.float32)
    buf93 = empty((768, ), device='cpu', dtype=torch.float32)
    buf94 = buf82; del buf82  # reuse
    buf95 = reinterpret_tensor(buf62, (6144, 196), (196, 1), 0); del buf62  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_14(c_void_p(buf94.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()))
    del div_8
    del mul_85
    del primals_105
    buf96 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf95, permute_141, out=buf96)
    del permute_141
    buf97 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (196, 6144), (1, 196), 0), view_67, out=buf97)
    del view_67
    buf98 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf99 = reinterpret_tensor(buf96, (8, 768, 384), (294912, 384, 1), 0); del buf96  # reuse
    buf100 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_15(c_void_p(buf99.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(mm_8.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()))
    del mm_8
    del primals_102
    buf101 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (384, 6144), (1, 384), 0), view_65, out=buf101)
    del view_65
    buf102 = buf95; del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (6144, 384), (384, 1), 0), permute_147, out=buf102)
    del permute_147
    buf103 = buf91; del buf91  # reuse
    buf104 = buf90; del buf90  # reuse
    buf105 = empty((768, ), device='cpu', dtype=torch.float32)
    buf106 = empty((768, ), device='cpu', dtype=torch.float32)
    buf107 = buf94; del buf94  # reuse
    cpp_fused_add_native_layer_norm_backward_16(c_void_p(buf107.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del div_9
    del mul_80
    del primals_99
    buf108 = reinterpret_tensor(buf86, (1568, 3072), (3072, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (1568, 768), (768, 1), 0), permute_150, out=buf108)
    del permute_150
    buf109 = reinterpret_tensor(buf99, (768, 3072), (3072, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (768, 1568), (1, 768), 0), view_63, out=buf109)
    del view_63
    buf110 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf111 = reinterpret_tensor(buf108, (8, 196, 3072), (602112, 3072, 1), 0); del buf108  # reuse
    cpp_fused_gelu_gelu_backward_sum_17(c_void_p(buf111.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf110.data_ptr()))
    del addmm_22
    buf112 = reinterpret_tensor(buf102, (1568, 768), (768, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1568, 3072), (3072, 1), 0), permute_154, out=buf112)
    del permute_154
    buf113 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (3072, 1568), (1, 3072), 0), view_61, out=buf113)
    del view_61
    buf114 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf115 = buf104; del buf104  # reuse
    buf116 = buf103; del buf103  # reuse
    buf117 = empty((768, ), device='cpu', dtype=torch.float32)
    buf118 = empty((768, ), device='cpu', dtype=torch.float32)
    buf119 = buf107; del buf107  # reuse
    buf120 = reinterpret_tensor(buf87, (6144, 196), (196, 1), 0); del buf87  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_18(c_void_p(buf119.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del div_10
    del mul_75
    del primals_93
    buf121 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf120, permute_159, out=buf121)
    del permute_159
    buf122 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (196, 6144), (1, 196), 0), view_59, out=buf122)
    del view_59
    buf123 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf121, (8, 768, 384), (294912, 384, 1), 0); del buf121  # reuse
    buf125 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_19(c_void_p(buf124.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(mm_7.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del mm_7
    del primals_90
    buf126 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (384, 6144), (1, 384), 0), view_57, out=buf126)
    del view_57
    buf127 = buf120; del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (6144, 384), (384, 1), 0), permute_165, out=buf127)
    del permute_165
    buf128 = buf116; del buf116  # reuse
    buf129 = buf115; del buf115  # reuse
    buf130 = empty((768, ), device='cpu', dtype=torch.float32)
    buf131 = empty((768, ), device='cpu', dtype=torch.float32)
    buf132 = buf119; del buf119  # reuse
    cpp_fused_add_native_layer_norm_backward_20(c_void_p(buf132.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(mul_70.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del div_11
    del mul_70
    del primals_87
    buf133 = reinterpret_tensor(buf111, (1568, 3072), (3072, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (1568, 768), (768, 1), 0), permute_168, out=buf133)
    del permute_168
    buf134 = reinterpret_tensor(buf124, (768, 3072), (3072, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (768, 1568), (1, 768), 0), view_55, out=buf134)
    del view_55
    buf135 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf136 = reinterpret_tensor(buf133, (8, 196, 3072), (602112, 3072, 1), 0); del buf133  # reuse
    cpp_fused_gelu_gelu_backward_sum_21(c_void_p(buf136.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf135.data_ptr()))
    del addmm_19
    buf137 = reinterpret_tensor(buf127, (1568, 768), (768, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (1568, 3072), (3072, 1), 0), permute_172, out=buf137)
    del permute_172
    buf138 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (3072, 1568), (1, 3072), 0), view_53, out=buf138)
    del view_53
    buf139 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf140 = buf129; del buf129  # reuse
    buf141 = buf128; del buf128  # reuse
    buf142 = empty((768, ), device='cpu', dtype=torch.float32)
    buf143 = empty((768, ), device='cpu', dtype=torch.float32)
    buf144 = buf132; del buf132  # reuse
    buf145 = reinterpret_tensor(buf112, (6144, 196), (196, 1), 0); del buf112  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_22(c_void_p(buf144.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()))
    del div_12
    del mul_65
    del primals_81
    buf146 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf145, permute_177, out=buf146)
    del permute_177
    buf147 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (196, 6144), (1, 196), 0), view_51, out=buf147)
    del view_51
    buf148 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf149 = reinterpret_tensor(buf146, (8, 768, 384), (294912, 384, 1), 0); del buf146  # reuse
    buf150 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_23(c_void_p(buf149.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(mm_6.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    del mm_6
    del primals_78
    buf151 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (384, 6144), (1, 384), 0), view_49, out=buf151)
    del view_49
    buf152 = buf145; del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (6144, 384), (384, 1), 0), permute_183, out=buf152)
    del permute_183
    buf153 = buf141; del buf141  # reuse
    buf154 = buf140; del buf140  # reuse
    buf155 = empty((768, ), device='cpu', dtype=torch.float32)
    buf156 = empty((768, ), device='cpu', dtype=torch.float32)
    buf157 = buf144; del buf144  # reuse
    cpp_fused_add_native_layer_norm_backward_24(c_void_p(buf157.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del div_13
    del mul_60
    del primals_75
    buf158 = reinterpret_tensor(buf136, (1568, 3072), (3072, 1), 0); del buf136  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (1568, 768), (768, 1), 0), permute_186, out=buf158)
    del permute_186
    buf159 = reinterpret_tensor(buf149, (768, 3072), (3072, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (768, 1568), (1, 768), 0), view_47, out=buf159)
    del view_47
    buf160 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf161 = reinterpret_tensor(buf158, (8, 196, 3072), (602112, 3072, 1), 0); del buf158  # reuse
    cpp_fused_gelu_gelu_backward_sum_25(c_void_p(buf161.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf160.data_ptr()))
    del addmm_16
    buf162 = reinterpret_tensor(buf152, (1568, 768), (768, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1568, 3072), (3072, 1), 0), permute_190, out=buf162)
    del permute_190
    buf163 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (3072, 1568), (1, 3072), 0), view_45, out=buf163)
    del view_45
    buf164 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf165 = buf154; del buf154  # reuse
    buf166 = buf153; del buf153  # reuse
    buf167 = empty((768, ), device='cpu', dtype=torch.float32)
    buf168 = empty((768, ), device='cpu', dtype=torch.float32)
    buf169 = buf157; del buf157  # reuse
    buf170 = reinterpret_tensor(buf137, (6144, 196), (196, 1), 0); del buf137  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_26(c_void_p(buf169.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(mul_55.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del div_14
    del mul_55
    del primals_69
    buf171 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf170, permute_195, out=buf171)
    del permute_195
    buf172 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf170, (196, 6144), (1, 196), 0), view_43, out=buf172)
    del view_43
    buf173 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf171, (8, 768, 384), (294912, 384, 1), 0); del buf171  # reuse
    buf175 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_27(c_void_p(buf174.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()))
    del mm_5
    del primals_66
    buf176 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (384, 6144), (1, 384), 0), view_41, out=buf176)
    del view_41
    buf177 = buf170; del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (6144, 384), (384, 1), 0), permute_201, out=buf177)
    del permute_201
    buf178 = buf166; del buf166  # reuse
    buf179 = buf165; del buf165  # reuse
    buf180 = empty((768, ), device='cpu', dtype=torch.float32)
    buf181 = empty((768, ), device='cpu', dtype=torch.float32)
    buf182 = buf169; del buf169  # reuse
    cpp_fused_add_native_layer_norm_backward_28(c_void_p(buf182.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del div_15
    del mul_50
    del primals_63
    buf183 = reinterpret_tensor(buf161, (1568, 3072), (3072, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (1568, 768), (768, 1), 0), permute_204, out=buf183)
    del permute_204
    buf184 = reinterpret_tensor(buf174, (768, 3072), (3072, 1), 0); del buf174  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (768, 1568), (1, 768), 0), view_39, out=buf184)
    del view_39
    buf185 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf186 = reinterpret_tensor(buf183, (8, 196, 3072), (602112, 3072, 1), 0); del buf183  # reuse
    cpp_fused_gelu_gelu_backward_sum_29(c_void_p(buf186.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf185.data_ptr()))
    del addmm_13
    buf187 = reinterpret_tensor(buf177, (1568, 768), (768, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (1568, 3072), (3072, 1), 0), permute_208, out=buf187)
    del permute_208
    buf188 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (3072, 1568), (1, 3072), 0), view_37, out=buf188)
    del view_37
    buf189 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf190 = buf179; del buf179  # reuse
    buf191 = buf178; del buf178  # reuse
    buf192 = empty((768, ), device='cpu', dtype=torch.float32)
    buf193 = empty((768, ), device='cpu', dtype=torch.float32)
    buf194 = buf182; del buf182  # reuse
    buf195 = reinterpret_tensor(buf162, (6144, 196), (196, 1), 0); del buf162  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_30(c_void_p(buf194.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del div_16
    del mul_45
    del primals_57
    buf196 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf195, permute_213, out=buf196)
    del permute_213
    buf197 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (196, 6144), (1, 196), 0), view_35, out=buf197)
    del view_35
    buf198 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf196, (8, 768, 384), (294912, 384, 1), 0); del buf196  # reuse
    buf200 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_31(c_void_p(buf199.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(mm_4.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    del mm_4
    del primals_54
    buf201 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (384, 6144), (1, 384), 0), view_33, out=buf201)
    del view_33
    buf202 = buf195; del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (6144, 384), (384, 1), 0), permute_219, out=buf202)
    del permute_219
    buf203 = buf191; del buf191  # reuse
    buf204 = buf190; del buf190  # reuse
    buf205 = empty((768, ), device='cpu', dtype=torch.float32)
    buf206 = empty((768, ), device='cpu', dtype=torch.float32)
    buf207 = buf194; del buf194  # reuse
    cpp_fused_add_native_layer_norm_backward_32(c_void_p(buf207.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del div_17
    del mul_40
    del primals_51
    buf208 = reinterpret_tensor(buf186, (1568, 3072), (3072, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (1568, 768), (768, 1), 0), permute_222, out=buf208)
    del permute_222
    buf209 = reinterpret_tensor(buf199, (768, 3072), (3072, 1), 0); del buf199  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (768, 1568), (1, 768), 0), view_31, out=buf209)
    del view_31
    buf210 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf211 = reinterpret_tensor(buf208, (8, 196, 3072), (602112, 3072, 1), 0); del buf208  # reuse
    cpp_fused_gelu_gelu_backward_sum_33(c_void_p(buf211.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf210.data_ptr()))
    del addmm_10
    buf212 = reinterpret_tensor(buf202, (1568, 768), (768, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (1568, 3072), (3072, 1), 0), permute_226, out=buf212)
    del permute_226
    buf213 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (3072, 1568), (1, 3072), 0), view_29, out=buf213)
    del view_29
    buf214 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf215 = buf204; del buf204  # reuse
    buf216 = buf203; del buf203  # reuse
    buf217 = empty((768, ), device='cpu', dtype=torch.float32)
    buf218 = empty((768, ), device='cpu', dtype=torch.float32)
    buf219 = buf207; del buf207  # reuse
    buf220 = reinterpret_tensor(buf187, (6144, 196), (196, 1), 0); del buf187  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_34(c_void_p(buf219.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()))
    del div_18
    del mul_35
    del primals_45
    buf221 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf220, permute_231, out=buf221)
    del permute_231
    buf222 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (196, 6144), (1, 196), 0), view_27, out=buf222)
    del view_27
    buf223 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf224 = reinterpret_tensor(buf221, (8, 768, 384), (294912, 384, 1), 0); del buf221  # reuse
    buf225 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_35(c_void_p(buf224.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()))
    del mm_3
    del primals_42
    buf226 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (384, 6144), (1, 384), 0), view_25, out=buf226)
    del view_25
    buf227 = buf220; del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (6144, 384), (384, 1), 0), permute_237, out=buf227)
    del permute_237
    buf228 = buf216; del buf216  # reuse
    buf229 = buf215; del buf215  # reuse
    buf230 = empty((768, ), device='cpu', dtype=torch.float32)
    buf231 = empty((768, ), device='cpu', dtype=torch.float32)
    buf232 = buf219; del buf219  # reuse
    cpp_fused_add_native_layer_norm_backward_36(c_void_p(buf232.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del div_19
    del mul_30
    del primals_39
    buf233 = reinterpret_tensor(buf211, (1568, 3072), (3072, 1), 0); del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (1568, 768), (768, 1), 0), permute_240, out=buf233)
    del permute_240
    buf234 = reinterpret_tensor(buf224, (768, 3072), (3072, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (768, 1568), (1, 768), 0), view_23, out=buf234)
    del view_23
    buf235 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf236 = reinterpret_tensor(buf233, (8, 196, 3072), (602112, 3072, 1), 0); del buf233  # reuse
    cpp_fused_gelu_gelu_backward_sum_37(c_void_p(buf236.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf235.data_ptr()))
    del addmm_7
    buf237 = reinterpret_tensor(buf227, (1568, 768), (768, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (1568, 3072), (3072, 1), 0), permute_244, out=buf237)
    del permute_244
    buf238 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (3072, 1568), (1, 3072), 0), view_21, out=buf238)
    del view_21
    buf239 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf240 = buf229; del buf229  # reuse
    buf241 = buf228; del buf228  # reuse
    buf242 = empty((768, ), device='cpu', dtype=torch.float32)
    buf243 = empty((768, ), device='cpu', dtype=torch.float32)
    buf244 = buf232; del buf232  # reuse
    buf245 = reinterpret_tensor(buf212, (6144, 196), (196, 1), 0); del buf212  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_38(c_void_p(buf244.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del div_20
    del mul_25
    del primals_33
    buf246 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf245, permute_249, out=buf246)
    del permute_249
    buf247 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (196, 6144), (1, 196), 0), view_19, out=buf247)
    del view_19
    buf248 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf249 = reinterpret_tensor(buf246, (8, 768, 384), (294912, 384, 1), 0); del buf246  # reuse
    buf250 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_39(c_void_p(buf249.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(mm_2.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    del mm_2
    del primals_30
    buf251 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (384, 6144), (1, 384), 0), view_17, out=buf251)
    del view_17
    buf252 = buf245; del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (6144, 384), (384, 1), 0), permute_255, out=buf252)
    del permute_255
    buf253 = buf241; del buf241  # reuse
    buf254 = buf240; del buf240  # reuse
    buf255 = empty((768, ), device='cpu', dtype=torch.float32)
    buf256 = empty((768, ), device='cpu', dtype=torch.float32)
    buf257 = buf244; del buf244  # reuse
    cpp_fused_add_native_layer_norm_backward_40(c_void_p(buf257.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del div_21
    del mul_20
    del primals_27
    buf258 = reinterpret_tensor(buf236, (1568, 3072), (3072, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1568, 768), (768, 1), 0), permute_258, out=buf258)
    del permute_258
    buf259 = reinterpret_tensor(buf249, (768, 3072), (3072, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (768, 1568), (1, 768), 0), view_15, out=buf259)
    del view_15
    buf260 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf258, (8, 196, 3072), (602112, 3072, 1), 0); del buf258  # reuse
    cpp_fused_gelu_gelu_backward_sum_41(c_void_p(buf261.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf260.data_ptr()))
    del addmm_4
    buf262 = reinterpret_tensor(buf252, (1568, 768), (768, 1), 0); del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (1568, 3072), (3072, 1), 0), permute_262, out=buf262)
    del permute_262
    buf263 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (3072, 1568), (1, 3072), 0), view_13, out=buf263)
    del view_13
    buf264 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf265 = buf254; del buf254  # reuse
    buf266 = buf253; del buf253  # reuse
    buf267 = empty((768, ), device='cpu', dtype=torch.float32)
    buf268 = empty((768, ), device='cpu', dtype=torch.float32)
    buf269 = buf257; del buf257  # reuse
    buf270 = reinterpret_tensor(buf237, (6144, 196), (196, 1), 0); del buf237  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_42(c_void_p(buf269.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()))
    del div_22
    del mul_15
    del primals_21
    buf271 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf270, permute_267, out=buf271)
    del permute_267
    buf272 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (196, 6144), (1, 196), 0), view_11, out=buf272)
    del view_11
    buf273 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf274 = reinterpret_tensor(buf271, (8, 768, 384), (294912, 384, 1), 0); del buf271  # reuse
    buf275 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_43(c_void_p(buf274.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(mm_1.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()))
    del mm_1
    del primals_18
    buf276 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (384, 6144), (1, 384), 0), view_9, out=buf276)
    del view_9
    buf277 = buf270; del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (6144, 384), (384, 1), 0), permute_273, out=buf277)
    del permute_273
    buf278 = buf266; del buf266  # reuse
    buf279 = buf265; del buf265  # reuse
    buf280 = empty((768, ), device='cpu', dtype=torch.float32)
    buf281 = empty((768, ), device='cpu', dtype=torch.float32)
    buf282 = buf269; del buf269  # reuse
    cpp_fused_add_native_layer_norm_backward_44(c_void_p(buf282.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del div_23
    del mul_10
    del primals_15
    buf283 = reinterpret_tensor(buf261, (1568, 3072), (3072, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (1568, 768), (768, 1), 0), permute_276, out=buf283)
    del permute_276
    buf284 = reinterpret_tensor(buf274, (768, 3072), (3072, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (768, 1568), (1, 768), 0), view_7, out=buf284)
    del view_7
    buf285 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf286 = reinterpret_tensor(buf283, (8, 196, 3072), (602112, 3072, 1), 0); del buf283  # reuse
    cpp_fused_gelu_gelu_backward_sum_45(c_void_p(buf286.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf285.data_ptr()))
    del addmm_1
    buf287 = reinterpret_tensor(buf277, (1568, 768), (768, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (1568, 3072), (3072, 1), 0), permute_280, out=buf287)
    del permute_280
    buf288 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (3072, 1568), (1, 3072), 0), view_5, out=buf288)
    del view_5
    buf289 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf290 = buf279; del buf279  # reuse
    buf291 = buf278; del buf278  # reuse
    buf292 = empty((768, ), device='cpu', dtype=torch.float32)
    buf293 = empty((768, ), device='cpu', dtype=torch.float32)
    buf294 = buf282; del buf282  # reuse
    buf295 = reinterpret_tensor(buf262, (6144, 196), (196, 1), 0); del buf262  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_46(c_void_p(buf294.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(mul_5.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()))
    del buf286
    del buf287
    del div_24
    del mul_5
    del primals_9
    buf296 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf295, permute_285, out=buf296)
    del permute_285
    buf297 = empty((196, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (196, 6144), (1, 196), 0), view_3, out=buf297)
    del view_3
    buf298 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf296, (8, 768, 384), (294912, 384, 1), 0); del buf296  # reuse
    buf300 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_sum_47(c_void_p(buf299.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(mm.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()))
    del mm
    del primals_6
    buf301 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (384, 6144), (1, 384), 0), view_1, out=buf301)
    del view_1
    buf302 = buf295; del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (6144, 384), (384, 1), 0), permute_291, out=buf302)
    del buf299
    del permute_291
    buf303 = buf291; del buf291  # reuse
    buf304 = buf290; del buf290  # reuse
    buf305 = empty((768, ), device='cpu', dtype=torch.float32)
    buf306 = empty((768, ), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf294, (8, 768, 14, 14), (150528, 1, 10752, 768), 0); del buf294  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_48(c_void_p(buf307.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del buf302
    del buf303
    del buf304
    del div_25
    del mul
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf308 = aten.convolution_backward(buf307, primals_151, primals_1, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf307
    del primals_1
    del primals_151
    buf309 = buf308[1]
    buf310 = buf308[2]
    return (buf309, buf310, buf305, buf306, reinterpret_tensor(buf301, (384, 196), (196, 1), 0), reinterpret_tensor(buf300, (384, ), (1, ), 0), reinterpret_tensor(buf297, (196, 384), (384, 1), 0), reinterpret_tensor(buf298, (196, ), (1, ), 0), buf292, buf293, reinterpret_tensor(buf288, (3072, 768), (768, 1), 0), reinterpret_tensor(buf289, (3072, ), (1, ), 0), reinterpret_tensor(buf284, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf285, (768, ), (1, ), 0), buf280, buf281, reinterpret_tensor(buf276, (384, 196), (196, 1), 0), reinterpret_tensor(buf275, (384, ), (1, ), 0), reinterpret_tensor(buf272, (196, 384), (384, 1), 0), reinterpret_tensor(buf273, (196, ), (1, ), 0), buf267, buf268, reinterpret_tensor(buf263, (3072, 768), (768, 1), 0), reinterpret_tensor(buf264, (3072, ), (1, ), 0), reinterpret_tensor(buf259, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf260, (768, ), (1, ), 0), buf255, buf256, reinterpret_tensor(buf251, (384, 196), (196, 1), 0), reinterpret_tensor(buf250, (384, ), (1, ), 0), reinterpret_tensor(buf247, (196, 384), (384, 1), 0), reinterpret_tensor(buf248, (196, ), (1, ), 0), buf242, buf243, reinterpret_tensor(buf238, (3072, 768), (768, 1), 0), reinterpret_tensor(buf239, (3072, ), (1, ), 0), reinterpret_tensor(buf234, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf235, (768, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf226, (384, 196), (196, 1), 0), reinterpret_tensor(buf225, (384, ), (1, ), 0), reinterpret_tensor(buf222, (196, 384), (384, 1), 0), reinterpret_tensor(buf223, (196, ), (1, ), 0), buf217, buf218, reinterpret_tensor(buf213, (3072, 768), (768, 1), 0), reinterpret_tensor(buf214, (3072, ), (1, ), 0), reinterpret_tensor(buf209, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), buf205, buf206, reinterpret_tensor(buf201, (384, 196), (196, 1), 0), reinterpret_tensor(buf200, (384, ), (1, ), 0), reinterpret_tensor(buf197, (196, 384), (384, 1), 0), reinterpret_tensor(buf198, (196, ), (1, ), 0), buf192, buf193, reinterpret_tensor(buf188, (3072, 768), (768, 1), 0), reinterpret_tensor(buf189, (3072, ), (1, ), 0), reinterpret_tensor(buf184, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf185, (768, ), (1, ), 0), buf180, buf181, reinterpret_tensor(buf176, (384, 196), (196, 1), 0), reinterpret_tensor(buf175, (384, ), (1, ), 0), reinterpret_tensor(buf172, (196, 384), (384, 1), 0), reinterpret_tensor(buf173, (196, ), (1, ), 0), buf167, buf168, reinterpret_tensor(buf163, (3072, 768), (768, 1), 0), reinterpret_tensor(buf164, (3072, ), (1, ), 0), reinterpret_tensor(buf159, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf160, (768, ), (1, ), 0), buf155, buf156, reinterpret_tensor(buf151, (384, 196), (196, 1), 0), reinterpret_tensor(buf150, (384, ), (1, ), 0), reinterpret_tensor(buf147, (196, 384), (384, 1), 0), reinterpret_tensor(buf148, (196, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf138, (3072, 768), (768, 1), 0), reinterpret_tensor(buf139, (3072, ), (1, ), 0), reinterpret_tensor(buf134, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf135, (768, ), (1, ), 0), buf130, buf131, reinterpret_tensor(buf126, (384, 196), (196, 1), 0), reinterpret_tensor(buf125, (384, ), (1, ), 0), reinterpret_tensor(buf122, (196, 384), (384, 1), 0), reinterpret_tensor(buf123, (196, ), (1, ), 0), buf117, buf118, reinterpret_tensor(buf113, (3072, 768), (768, 1), 0), reinterpret_tensor(buf114, (3072, ), (1, ), 0), reinterpret_tensor(buf109, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), buf105, buf106, reinterpret_tensor(buf101, (384, 196), (196, 1), 0), reinterpret_tensor(buf100, (384, ), (1, ), 0), reinterpret_tensor(buf97, (196, 384), (384, 1), 0), reinterpret_tensor(buf98, (196, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf88, (3072, 768), (768, 1), 0), reinterpret_tensor(buf89, (3072, ), (1, ), 0), reinterpret_tensor(buf84, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf85, (768, ), (1, ), 0), buf80, buf81, reinterpret_tensor(buf76, (384, 196), (196, 1), 0), reinterpret_tensor(buf75, (384, ), (1, ), 0), reinterpret_tensor(buf72, (196, 384), (384, 1), 0), reinterpret_tensor(buf73, (196, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf63, (3072, 768), (768, 1), 0), reinterpret_tensor(buf64, (3072, ), (1, ), 0), reinterpret_tensor(buf59, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf60, (768, ), (1, ), 0), buf55, buf56, reinterpret_tensor(buf51, (384, 196), (196, 1), 0), reinterpret_tensor(buf50, (384, ), (1, ), 0), reinterpret_tensor(buf47, (196, 384), (384, 1), 0), reinterpret_tensor(buf48, (196, ), (1, ), 0), buf42, buf43, reinterpret_tensor(buf38, (3072, 768), (768, 1), 0), reinterpret_tensor(buf39, (3072, ), (1, ), 0), reinterpret_tensor(buf34, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf35, (768, ), (1, ), 0), buf30, buf31, reinterpret_tensor(buf26, (384, 196), (196, 1), 0), reinterpret_tensor(buf25, (384, ), (1, ), 0), reinterpret_tensor(buf22, (196, 384), (384, 1), 0), reinterpret_tensor(buf23, (196, ), (1, ), 0), buf17, buf18, reinterpret_tensor(buf13, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf9, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf10, (768, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_10 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_1 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_2 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_25 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_3 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_33 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_4 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_6 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_70 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_7 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_8 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_9 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_95 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_100 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_10 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_105 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_110 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((6144, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_115 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    clone_85 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_78 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_93 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_96 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_111 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_129 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_276 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, mul, view_1, mm, view_3, mul_5, view_5, addmm_1, view_7, mul_10, view_9, mm_1, view_11, mul_15, view_13, addmm_4, view_15, mul_20, view_17, mm_2, view_19, mul_25, view_21, addmm_7, view_23, mul_30, view_25, mm_3, view_27, mul_35, view_29, addmm_10, view_31, mul_40, view_33, mm_4, view_35, mul_45, view_37, addmm_13, view_39, mul_50, view_41, mm_5, view_43, mul_55, view_45, addmm_16, view_47, mul_60, view_49, mm_6, view_51, mul_65, view_53, addmm_19, view_55, mul_70, view_57, mm_7, view_59, mul_75, view_61, addmm_22, view_63, mul_80, view_65, mm_8, view_67, mul_85, view_69, addmm_25, view_71, mul_90, view_73, mm_9, view_75, mul_95, view_77, addmm_28, view_79, mul_100, view_81, mm_10, view_83, mul_105, view_85, addmm_31, view_87, mul_110, view_89, mm_11, view_91, mul_115, view_93, addmm_34, view_95, mul_120, clone_85, permute_74, div_1, permute_78, permute_82, div_2, permute_87, permute_93, div_3, permute_96, permute_100, div_4, permute_105, permute_111, div_5, permute_114, permute_118, div_6, permute_123, permute_129, div_7, permute_132, permute_136, div_8, permute_141, permute_147, div_9, permute_150, permute_154, div_10, permute_159, permute_165, div_11, permute_168, permute_172, div_12, permute_177, permute_183, div_13, permute_186, permute_190, div_14, permute_195, permute_201, div_15, permute_204, permute_208, div_16, permute_213, permute_219, div_17, permute_222, permute_226, div_18, permute_231, permute_237, div_19, permute_240, permute_244, div_20, permute_249, permute_255, div_21, permute_258, permute_262, div_22, permute_267, permute_273, div_23, permute_276, permute_280, div_24, permute_285, permute_291, div_25, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
