
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


cpp_fused_gelu_gelu_backward_sum_0 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_7 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_8 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_15 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_23 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_32 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_39 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_40 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_48 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_64 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_71 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_72 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_80 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_87 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    tmp21.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_88 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_eq_masked_fill_mul_neg_sum_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = in_ptr2[static_cast<long>(x1)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = static_cast<float>(1e-06);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 / tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = tmp0.neg();
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp1 * tmp10;
                        auto tmp12 = tmp0 / tmp8;
                        auto tmp13 = tmp12 * tmp2;
                        auto tmp14 = tmp13.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr3[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp23 = out_ptr4[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 / tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp0 + tmp8;
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 == tmp10;
                    auto tmp13 = static_cast<float>(2.0);
                    auto tmp14 = decltype(tmp2)(tmp2 * tmp13);
                    auto tmp15 = tmp12 / tmp14;
                    auto tmp16 = tmp11 ? tmp10 : tmp15;
                    auto tmp17 = static_cast<float>(0.002607561929595828);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp20 = at::vec::Vectorized<float>(tmp18);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp9 + tmp21;
                    auto tmp24 = static_cast<float>(768.0);
                    auto tmp25 = tmp23 / tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp22 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_eq_masked_fill_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp8 = decltype(tmp5)(tmp5 * tmp7);
                            auto tmp9 = decltype(tmp6)(tmp6 - tmp8);
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp3 ? tmp10 : tmp9;
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((128L*x1) + (128L*x1_inner) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>(x0) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_eq_masked_fill_mul_neg_sum_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const long* in_ptr7,
                       const long* in_ptr8,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = static_cast<float>(1e-06);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp4 / tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp8 / tmp12;
                        auto tmp14 = tmp13 / tmp12;
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp4 / tmp12;
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp18 = tmp17.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr4[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp17 = out_ptr5[static_cast<long>(x0)];
                    auto tmp22 = in_ptr7[static_cast<long>(x0)];
                    auto tmp28 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 == tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = tmp5 ? tmp4 : tmp9;
                    auto tmp11 = static_cast<float>(0.002607561929595828);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = at::vec::Vectorized<float>(tmp12);
                    auto tmp15 = tmp14 * tmp13;
                    auto tmp16 = tmp2 + tmp15;
                    auto tmp18 = static_cast<float>(768.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp16 + tmp20;
                    auto tmp23 = static_cast<int>(0);
                    auto tmp24 = tmp22 == tmp23;
                    auto tmp25 = to_float_mask(tmp24);
                    auto tmp26 = at::vec::Vectorized<float>(tmp4);
                    auto tmp27 = decltype(tmp26)::blendv(tmp21, tmp26, tmp25);
                    auto tmp29 = tmp28 == tmp23;
                    auto tmp30 = to_float_mask(tmp29);
                    auto tmp31 = decltype(tmp26)::blendv(tmp21, tmp26, tmp30);
                    tmp27.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    tmp31.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(15363840L); x0+=static_cast<long>(8L))
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, unsqueeze_1, sqrt, sub, view, view_16, sqrt_1, sub_2, view_18, addmm_4, view_20, sqrt_2, sub_3, view_22, view_38, sqrt_3, sub_5, view_40, addmm_10, view_42, sqrt_4, sub_6, view_44, view_60, sqrt_5, sub_8, view_62, addmm_16, view_64, sqrt_6, sub_9, view_66, view_82, sqrt_7, sub_11, view_84, addmm_22, view_86, sqrt_8, sub_12, view_88, view_104, sqrt_9, sub_14, view_106, addmm_28, view_108, sqrt_10, sub_15, view_110, view_126, sqrt_11, sub_17, view_128, addmm_34, view_130, sqrt_12, sub_18, view_132, view_148, sqrt_13, sub_20, view_150, addmm_40, view_152, sqrt_14, sub_21, view_154, view_170, sqrt_15, sub_23, view_172, addmm_46, view_174, sqrt_16, sub_24, view_176, view_192, sqrt_17, sub_26, view_194, addmm_52, view_196, sqrt_18, sub_27, view_198, view_214, sqrt_19, sub_29, view_216, addmm_58, view_218, sqrt_20, sub_30, view_220, view_236, sqrt_21, sub_32, view_238, addmm_64, view_240, sqrt_22, sub_33, view_242, view_258, sqrt_23, sub_35, view_260, addmm_70, view_262, permute_132, permute_136, permute_140, permute_145, permute_146, alias_37, permute_147, permute_148, permute_151, permute_156, permute_161, permute_165, permute_169, permute_173, permute_178, permute_179, alias_40, permute_180, permute_181, permute_184, permute_189, permute_194, permute_198, permute_202, permute_206, permute_211, permute_212, alias_43, permute_213, permute_214, permute_217, permute_222, permute_227, permute_231, permute_235, permute_239, permute_244, permute_245, alias_46, permute_246, permute_247, permute_250, permute_255, permute_260, permute_264, permute_268, permute_272, permute_277, permute_278, alias_49, permute_279, permute_280, permute_283, permute_288, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, alias_52, permute_312, permute_313, permute_316, permute_321, permute_326, permute_330, permute_334, permute_338, permute_343, permute_344, alias_55, permute_345, permute_346, permute_349, permute_354, permute_359, permute_363, permute_367, permute_371, permute_376, permute_377, alias_58, permute_378, permute_379, permute_382, permute_387, permute_392, permute_396, permute_400, permute_404, permute_409, permute_410, alias_61, permute_411, permute_412, permute_415, permute_420, permute_425, permute_429, permute_433, permute_437, permute_442, permute_443, alias_64, permute_444, permute_445, permute_448, permute_453, permute_458, permute_462, permute_466, permute_470, permute_475, permute_476, alias_67, permute_477, permute_478, permute_481, permute_486, permute_491, permute_495, permute_499, permute_503, permute_508, permute_509, alias_70, permute_510, permute_511, permute_514, permute_519, permute_524, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_196, (4, 128), (128, 1))
    assert_size_stride(primals_197, (4, 128), (128, 1))
    assert_size_stride(unsqueeze_1, (4, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(sqrt, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(sqrt_1, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_2, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_2, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_3, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(sqrt_3, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_5, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_4, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_6, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(sqrt_5, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_8, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_6, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_9, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(sqrt_7, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_11, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_8, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_12, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(sqrt_9, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_14, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_10, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_15, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(sqrt_11, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_17, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_12, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_18, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(sqrt_13, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_20, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_14, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_21, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(sqrt_15, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_23, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_16, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_24, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(sqrt_17, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_26, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_18, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_27, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(sqrt_19, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_29, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_20, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_30, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(sqrt_21, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_32, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(sqrt_22, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_33, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(sqrt_23, (4, 128, 1), (128, 1, 1))
    assert_size_stride(sub_35, (4, 128, 768), (98304, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(permute_140, (768, 768), (768, 1))
    assert_size_stride(permute_145, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_146, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_37, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_147, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_148, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_156, (768, 768), (768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_165, (768, 3072), (3072, 1))
    assert_size_stride(permute_169, (3072, 768), (768, 1))
    assert_size_stride(permute_173, (768, 768), (768, 1))
    assert_size_stride(permute_178, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_179, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_40, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_180, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_181, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_184, (768, 768), (768, 1))
    assert_size_stride(permute_189, (768, 768), (768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_198, (768, 3072), (3072, 1))
    assert_size_stride(permute_202, (3072, 768), (768, 1))
    assert_size_stride(permute_206, (768, 768), (768, 1))
    assert_size_stride(permute_211, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_212, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_43, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_213, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_214, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_217, (768, 768), (768, 1))
    assert_size_stride(permute_222, (768, 768), (768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_231, (768, 3072), (3072, 1))
    assert_size_stride(permute_235, (3072, 768), (768, 1))
    assert_size_stride(permute_239, (768, 768), (768, 1))
    assert_size_stride(permute_244, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_245, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_46, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_246, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_247, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_250, (768, 768), (768, 1))
    assert_size_stride(permute_255, (768, 768), (768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_264, (768, 3072), (3072, 1))
    assert_size_stride(permute_268, (3072, 768), (768, 1))
    assert_size_stride(permute_272, (768, 768), (768, 1))
    assert_size_stride(permute_277, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_278, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_49, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_279, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_280, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_283, (768, 768), (768, 1))
    assert_size_stride(permute_288, (768, 768), (768, 1))
    assert_size_stride(permute_293, (768, 768), (768, 1))
    assert_size_stride(permute_297, (768, 3072), (3072, 1))
    assert_size_stride(permute_301, (3072, 768), (768, 1))
    assert_size_stride(permute_305, (768, 768), (768, 1))
    assert_size_stride(permute_310, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_311, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_52, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_312, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_313, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_316, (768, 768), (768, 1))
    assert_size_stride(permute_321, (768, 768), (768, 1))
    assert_size_stride(permute_326, (768, 768), (768, 1))
    assert_size_stride(permute_330, (768, 3072), (3072, 1))
    assert_size_stride(permute_334, (3072, 768), (768, 1))
    assert_size_stride(permute_338, (768, 768), (768, 1))
    assert_size_stride(permute_343, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_344, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_55, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_345, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_346, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_349, (768, 768), (768, 1))
    assert_size_stride(permute_354, (768, 768), (768, 1))
    assert_size_stride(permute_359, (768, 768), (768, 1))
    assert_size_stride(permute_363, (768, 3072), (3072, 1))
    assert_size_stride(permute_367, (3072, 768), (768, 1))
    assert_size_stride(permute_371, (768, 768), (768, 1))
    assert_size_stride(permute_376, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_377, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_58, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_378, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_379, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_382, (768, 768), (768, 1))
    assert_size_stride(permute_387, (768, 768), (768, 1))
    assert_size_stride(permute_392, (768, 768), (768, 1))
    assert_size_stride(permute_396, (768, 3072), (3072, 1))
    assert_size_stride(permute_400, (3072, 768), (768, 1))
    assert_size_stride(permute_404, (768, 768), (768, 1))
    assert_size_stride(permute_409, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_410, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_61, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_411, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_412, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_415, (768, 768), (768, 1))
    assert_size_stride(permute_420, (768, 768), (768, 1))
    assert_size_stride(permute_425, (768, 768), (768, 1))
    assert_size_stride(permute_429, (768, 3072), (3072, 1))
    assert_size_stride(permute_433, (3072, 768), (768, 1))
    assert_size_stride(permute_437, (768, 768), (768, 1))
    assert_size_stride(permute_442, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_443, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_64, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_444, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_445, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_448, (768, 768), (768, 1))
    assert_size_stride(permute_453, (768, 768), (768, 1))
    assert_size_stride(permute_458, (768, 768), (768, 1))
    assert_size_stride(permute_462, (768, 3072), (3072, 1))
    assert_size_stride(permute_466, (3072, 768), (768, 1))
    assert_size_stride(permute_470, (768, 768), (768, 1))
    assert_size_stride(permute_475, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_476, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_67, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_477, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_478, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_481, (768, 768), (768, 1))
    assert_size_stride(permute_486, (768, 768), (768, 1))
    assert_size_stride(permute_491, (768, 768), (768, 1))
    assert_size_stride(permute_495, (768, 3072), (3072, 1))
    assert_size_stride(permute_499, (3072, 768), (768, 1))
    assert_size_stride(permute_503, (768, 768), (768, 1))
    assert_size_stride(permute_508, (48, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_509, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_70, (4, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_510, (48, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_511, (48, 128, 64), (8192, 1, 128))
    assert_size_stride(permute_514, (768, 768), (768, 1))
    assert_size_stride(permute_519, (768, 768), (768, 1))
    assert_size_stride(permute_524, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (4, 128, 768), (98304, 768, 1))
    buf0 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (512, 768), (768, 1), 0), permute_132, out=buf0)
    del permute_132
    buf1 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (768, 512), (1, 768), 0), view_262, out=buf1)
    del view_262
    buf2 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf0, (4, 128, 3072), (393216, 3072, 1), 0); del buf0  # reuse
    cpp_fused_gelu_gelu_backward_sum_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(buf2.data_ptr()))
    del addmm_70
    buf4 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (512, 3072), (3072, 1), 0), permute_136, out=buf4)
    del permute_136
    buf5 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (3072, 512), (1, 3072), 0), view_260, out=buf5)
    del view_260
    buf6 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf4, (4, 128, 768), (98304, 768, 1), 0); del buf4  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(sqrt_23.data_ptr()), c_void_p(sub_35.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()))
    del primals_47
    del sqrt_23
    del sub_35
    del tangents_1
    buf12 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), permute_140, out=buf12)
    del permute_140
    buf13 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (768, 512), (1, 768), 0), view_258, out=buf13)
    del view_258
    buf14 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf15 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_2(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf12, (48, 128, 64), (8192, 64, 1), 0); del buf12  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_145, reinterpret_tensor(buf15, (48, 128, 64), (8192, 64, 1), 0), out=buf16)
    del permute_145
    buf17 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf15, (48, 128, 64), (8192, 64, 1), 0), permute_146, out=buf17)
    del permute_146
    buf18 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf17, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf17  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_3(c_void_p(buf19.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf18.data_ptr()))
    del alias_37
    buf20 = reinterpret_tensor(buf15, (48, 64, 128), (8192, 128, 1), 0); del buf15  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_147, reinterpret_tensor(buf19, (48, 128, 128), (16384, 128, 1), 0), out=buf20)
    del permute_147
    buf21 = empty((48, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf19, (48, 128, 128), (16384, 128, 1), 0), permute_148, out=buf21)
    del permute_148
    buf22 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_4(c_void_p(buf16.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf16, (512, 768), (768, 1), 0); del buf16  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf22, permute_151, out=buf23)
    del permute_151
    buf24 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (768, 512), (1, 768), 0), view_242, out=buf24)
    buf25 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf26 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_sum_5(c_void_p(buf22.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    buf27 = buf22; del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf26, permute_156, out=buf27)
    del permute_156
    buf28 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (768, 512), (1, 768), 0), view_242, out=buf28)
    buf29 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf30 = reinterpret_tensor(buf20, (512, 768), (768, 1), 0); del buf20  # reuse
    cpp_fused_sum_view_6(c_void_p(buf26.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    buf31 = buf26; del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf30, permute_161, out=buf31)
    del permute_161
    buf32 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (768, 512), (1, 768), 0), view_242, out=buf32)
    del view_242
    buf33 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf34 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf37 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf35 = buf8; del buf8  # reuse
    buf36 = reinterpret_tensor(buf21, (4, 128, 768), (98304, 768, 1), 0); del buf21  # reuse
    buf38 = buf10; del buf10  # reuse
    buf39 = buf11; del buf11  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_7(c_void_p(buf39.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(sqrt_22.data_ptr()), c_void_p(sub_33.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_45
    del sqrt_22
    del sub_33
    buf40 = reinterpret_tensor(buf3, (512, 3072), (3072, 1), 0); del buf3  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (512, 768), (768, 1), 0), permute_165, out=buf40)
    del permute_165
    buf41 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (768, 512), (1, 768), 0), view_240, out=buf41)
    del view_240
    buf42 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf43 = reinterpret_tensor(buf40, (4, 128, 3072), (393216, 3072, 1), 0); del buf40  # reuse
    cpp_fused_gelu_gelu_backward_sum_8(c_void_p(buf43.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(buf42.data_ptr()))
    del addmm_64
    buf44 = reinterpret_tensor(buf36, (512, 768), (768, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf43, (512, 3072), (3072, 1), 0), permute_169, out=buf44)
    del permute_169
    buf45 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf43, (3072, 512), (1, 3072), 0), view_238, out=buf45)
    del view_238
    buf46 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf47 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf49 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf48 = buf38; del buf38  # reuse
    buf50 = buf35; del buf35  # reuse
    buf51 = buf39; del buf39  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_9(c_void_p(buf51.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(sqrt_21.data_ptr()), c_void_p(sub_32.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del primals_43
    del sqrt_21
    del sub_32
    buf52 = buf44; del buf44  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (512, 768), (768, 1), 0), permute_173, out=buf52)
    del permute_173
    buf53 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (768, 512), (1, 768), 0), view_236, out=buf53)
    del view_236
    buf54 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf55 = reinterpret_tensor(buf31, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf31  # reuse
    cpp_fused_clone_sum_10(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    buf56 = reinterpret_tensor(buf52, (48, 128, 64), (8192, 64, 1), 0); del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_178, reinterpret_tensor(buf55, (48, 128, 64), (8192, 64, 1), 0), out=buf56)
    del permute_178
    buf57 = reinterpret_tensor(buf19, (48, 128, 128), (16384, 128, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (48, 128, 64), (8192, 64, 1), 0), permute_179, out=buf57)
    del permute_179
    buf58 = buf18; del buf18  # reuse
    buf59 = reinterpret_tensor(buf57, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf57  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_11(c_void_p(buf59.data_ptr()), c_void_p(alias_40.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf58.data_ptr()))
    del alias_40
    buf60 = reinterpret_tensor(buf55, (48, 64, 128), (8192, 128, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_180, reinterpret_tensor(buf59, (48, 128, 128), (16384, 128, 1), 0), out=buf60)
    del permute_180
    buf61 = reinterpret_tensor(buf30, (48, 128, 64), (8192, 64, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf59, (48, 128, 128), (16384, 128, 1), 0), permute_181, out=buf61)
    del permute_181
    buf62 = buf27; del buf27  # reuse
    cpp_fused_view_12(c_void_p(buf56.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf56, (512, 768), (768, 1), 0); del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf62, permute_184, out=buf63)
    del permute_184
    buf64 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (768, 512), (1, 768), 0), view_220, out=buf64)
    buf65 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf66 = buf23; del buf23  # reuse
    cpp_fused__unsafe_view_clone_sum_13(c_void_p(buf62.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = buf62; del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf66, permute_189, out=buf67)
    del permute_189
    buf68 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (768, 512), (1, 768), 0), view_220, out=buf68)
    buf69 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf70 = reinterpret_tensor(buf60, (512, 768), (768, 1), 0); del buf60  # reuse
    cpp_fused_sum_view_14(c_void_p(buf66.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = buf66; del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, permute_194, out=buf71)
    del permute_194
    buf72 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (768, 512), (1, 768), 0), view_220, out=buf72)
    del view_220
    buf73 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf74 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf77 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf75 = buf50; del buf50  # reuse
    buf76 = reinterpret_tensor(buf61, (4, 128, 768), (98304, 768, 1), 0); del buf61  # reuse
    buf78 = buf48; del buf48  # reuse
    buf79 = buf51; del buf51  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_15(c_void_p(buf79.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(sqrt_20.data_ptr()), c_void_p(sub_30.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del primals_41
    del sqrt_20
    del sub_30
    buf80 = reinterpret_tensor(buf43, (512, 3072), (3072, 1), 0); del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (512, 768), (768, 1), 0), permute_198, out=buf80)
    del permute_198
    buf81 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (768, 512), (1, 768), 0), view_218, out=buf81)
    del view_218
    buf82 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf83 = reinterpret_tensor(buf80, (4, 128, 3072), (393216, 3072, 1), 0); del buf80  # reuse
    cpp_fused_gelu_gelu_backward_sum_16(c_void_p(buf83.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf82.data_ptr()))
    del addmm_58
    buf84 = reinterpret_tensor(buf76, (512, 768), (768, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (512, 3072), (3072, 1), 0), permute_202, out=buf84)
    del permute_202
    buf85 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (3072, 512), (1, 3072), 0), view_216, out=buf85)
    del view_216
    buf86 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf87 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf89 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf88 = buf78; del buf78  # reuse
    buf90 = buf75; del buf75  # reuse
    buf91 = buf79; del buf79  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_17(c_void_p(buf91.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(sqrt_19.data_ptr()), c_void_p(sub_29.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_39
    del sqrt_19
    del sub_29
    buf92 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (512, 768), (768, 1), 0), permute_206, out=buf92)
    del permute_206
    buf93 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (768, 512), (1, 768), 0), view_214, out=buf93)
    del view_214
    buf94 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf71, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf71  # reuse
    cpp_fused_clone_sum_18(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    buf96 = reinterpret_tensor(buf92, (48, 128, 64), (8192, 64, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_211, reinterpret_tensor(buf95, (48, 128, 64), (8192, 64, 1), 0), out=buf96)
    del permute_211
    buf97 = reinterpret_tensor(buf59, (48, 128, 128), (16384, 128, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (48, 128, 64), (8192, 64, 1), 0), permute_212, out=buf97)
    del permute_212
    buf98 = buf58; del buf58  # reuse
    buf99 = reinterpret_tensor(buf97, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf97  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_19(c_void_p(buf99.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del alias_43
    buf100 = reinterpret_tensor(buf95, (48, 64, 128), (8192, 128, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_213, reinterpret_tensor(buf99, (48, 128, 128), (16384, 128, 1), 0), out=buf100)
    del permute_213
    buf101 = reinterpret_tensor(buf70, (48, 128, 64), (8192, 64, 1), 0); del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf99, (48, 128, 128), (16384, 128, 1), 0), permute_214, out=buf101)
    del permute_214
    buf102 = buf67; del buf67  # reuse
    cpp_fused_view_20(c_void_p(buf96.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf102, permute_217, out=buf103)
    del permute_217
    buf104 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (768, 512), (1, 768), 0), view_198, out=buf104)
    buf105 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf106 = buf63; del buf63  # reuse
    cpp_fused__unsafe_view_clone_sum_21(c_void_p(buf102.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = buf102; del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf106, permute_222, out=buf107)
    del permute_222
    buf108 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (768, 512), (1, 768), 0), view_198, out=buf108)
    buf109 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf100, (512, 768), (768, 1), 0); del buf100  # reuse
    cpp_fused_sum_view_22(c_void_p(buf106.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = buf106; del buf106  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf110, permute_227, out=buf111)
    del permute_227
    buf112 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (768, 512), (1, 768), 0), view_198, out=buf112)
    del view_198
    buf113 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf114 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf117 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf115 = buf90; del buf90  # reuse
    buf116 = reinterpret_tensor(buf101, (4, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
    buf118 = buf88; del buf88  # reuse
    buf119 = buf116; del buf116  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_23(c_void_p(buf119.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(sqrt_18.data_ptr()), c_void_p(sub_27.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()))
    del primals_37
    del sqrt_18
    del sub_27
    buf120 = reinterpret_tensor(buf83, (512, 3072), (3072, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), permute_231, out=buf120)
    del permute_231
    buf121 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (768, 512), (1, 768), 0), view_196, out=buf121)
    del view_196
    buf122 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf123 = reinterpret_tensor(buf120, (4, 128, 3072), (393216, 3072, 1), 0); del buf120  # reuse
    cpp_fused_gelu_gelu_backward_sum_24(c_void_p(buf123.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf122.data_ptr()))
    del addmm_52
    buf124 = reinterpret_tensor(buf91, (512, 768), (768, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (512, 3072), (3072, 1), 0), permute_235, out=buf124)
    del permute_235
    buf125 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (3072, 512), (1, 3072), 0), view_194, out=buf125)
    del view_194
    buf126 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf127 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf129 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf128 = buf118; del buf118  # reuse
    buf130 = buf115; del buf115  # reuse
    buf131 = buf119; del buf119  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_25(c_void_p(buf131.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(sqrt_17.data_ptr()), c_void_p(sub_26.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del primals_35
    del sqrt_17
    del sub_26
    buf132 = buf124; del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (512, 768), (768, 1), 0), permute_239, out=buf132)
    del permute_239
    buf133 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (768, 512), (1, 768), 0), view_192, out=buf133)
    del view_192
    buf134 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf111, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf111  # reuse
    cpp_fused_clone_sum_26(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = reinterpret_tensor(buf132, (48, 128, 64), (8192, 64, 1), 0); del buf132  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_244, reinterpret_tensor(buf135, (48, 128, 64), (8192, 64, 1), 0), out=buf136)
    del permute_244
    buf137 = reinterpret_tensor(buf99, (48, 128, 128), (16384, 128, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf135, (48, 128, 64), (8192, 64, 1), 0), permute_245, out=buf137)
    del permute_245
    buf138 = buf98; del buf98  # reuse
    buf139 = reinterpret_tensor(buf137, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf137  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_27(c_void_p(buf139.data_ptr()), c_void_p(alias_46.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf138.data_ptr()))
    del alias_46
    buf140 = reinterpret_tensor(buf135, (48, 64, 128), (8192, 128, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_246, reinterpret_tensor(buf139, (48, 128, 128), (16384, 128, 1), 0), out=buf140)
    del permute_246
    buf141 = reinterpret_tensor(buf110, (48, 128, 64), (8192, 64, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (48, 128, 128), (16384, 128, 1), 0), permute_247, out=buf141)
    del permute_247
    buf142 = buf107; del buf107  # reuse
    cpp_fused_view_28(c_void_p(buf136.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = reinterpret_tensor(buf136, (512, 768), (768, 1), 0); del buf136  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf142, permute_250, out=buf143)
    del permute_250
    buf144 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (768, 512), (1, 768), 0), view_176, out=buf144)
    buf145 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf146 = buf103; del buf103  # reuse
    cpp_fused__unsafe_view_clone_sum_29(c_void_p(buf142.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    buf147 = buf142; del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf146, permute_255, out=buf147)
    del permute_255
    buf148 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (768, 512), (1, 768), 0), view_176, out=buf148)
    buf149 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf150 = reinterpret_tensor(buf140, (512, 768), (768, 1), 0); del buf140  # reuse
    cpp_fused_sum_view_30(c_void_p(buf146.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = buf146; del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf150, permute_260, out=buf151)
    del permute_260
    buf152 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (768, 512), (1, 768), 0), view_176, out=buf152)
    del view_176
    buf153 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf154 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf157 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf155 = buf130; del buf130  # reuse
    buf156 = reinterpret_tensor(buf141, (4, 128, 768), (98304, 768, 1), 0); del buf141  # reuse
    buf158 = buf128; del buf128  # reuse
    buf159 = buf131; del buf131  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_31(c_void_p(buf159.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(sqrt_16.data_ptr()), c_void_p(sub_24.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del primals_33
    del sqrt_16
    del sub_24
    buf160 = reinterpret_tensor(buf123, (512, 3072), (3072, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (512, 768), (768, 1), 0), permute_264, out=buf160)
    del permute_264
    buf161 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (768, 512), (1, 768), 0), view_174, out=buf161)
    del view_174
    buf162 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf160, (4, 128, 3072), (393216, 3072, 1), 0); del buf160  # reuse
    cpp_fused_gelu_gelu_backward_sum_32(c_void_p(buf163.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf162.data_ptr()))
    del addmm_46
    buf164 = reinterpret_tensor(buf156, (512, 768), (768, 1), 0); del buf156  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (512, 3072), (3072, 1), 0), permute_268, out=buf164)
    del permute_268
    buf165 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (3072, 512), (1, 3072), 0), view_172, out=buf165)
    del view_172
    buf166 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf167 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf169 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf168 = buf158; del buf158  # reuse
    buf170 = buf155; del buf155  # reuse
    buf171 = buf159; del buf159  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_33(c_void_p(buf171.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(sqrt_15.data_ptr()), c_void_p(sub_23.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del primals_31
    del sqrt_15
    del sub_23
    buf172 = buf164; del buf164  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (512, 768), (768, 1), 0), permute_272, out=buf172)
    del permute_272
    buf173 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (768, 512), (1, 768), 0), view_170, out=buf173)
    del view_170
    buf174 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf175 = reinterpret_tensor(buf151, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf151  # reuse
    cpp_fused_clone_sum_34(c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf172, (48, 128, 64), (8192, 64, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_277, reinterpret_tensor(buf175, (48, 128, 64), (8192, 64, 1), 0), out=buf176)
    del permute_277
    buf177 = reinterpret_tensor(buf139, (48, 128, 128), (16384, 128, 1), 0); del buf139  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf175, (48, 128, 64), (8192, 64, 1), 0), permute_278, out=buf177)
    del permute_278
    buf178 = buf138; del buf138  # reuse
    buf179 = reinterpret_tensor(buf177, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf177  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_35(c_void_p(buf179.data_ptr()), c_void_p(alias_49.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf178.data_ptr()))
    del alias_49
    buf180 = reinterpret_tensor(buf175, (48, 64, 128), (8192, 128, 1), 0); del buf175  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_279, reinterpret_tensor(buf179, (48, 128, 128), (16384, 128, 1), 0), out=buf180)
    del permute_279
    buf181 = reinterpret_tensor(buf150, (48, 128, 64), (8192, 64, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf179, (48, 128, 128), (16384, 128, 1), 0), permute_280, out=buf181)
    del permute_280
    buf182 = buf147; del buf147  # reuse
    cpp_fused_view_36(c_void_p(buf176.data_ptr()), c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf176, (512, 768), (768, 1), 0); del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf182, permute_283, out=buf183)
    del permute_283
    buf184 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (768, 512), (1, 768), 0), view_154, out=buf184)
    buf185 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf186 = buf143; del buf143  # reuse
    cpp_fused__unsafe_view_clone_sum_37(c_void_p(buf182.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = buf182; del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf186, permute_288, out=buf187)
    del permute_288
    buf188 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (768, 512), (1, 768), 0), view_154, out=buf188)
    buf189 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf180, (512, 768), (768, 1), 0); del buf180  # reuse
    cpp_fused_sum_view_38(c_void_p(buf186.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = buf186; del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf190, permute_293, out=buf191)
    del permute_293
    buf192 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (768, 512), (1, 768), 0), view_154, out=buf192)
    del view_154
    buf193 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf194 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf197 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf195 = buf170; del buf170  # reuse
    buf196 = reinterpret_tensor(buf181, (4, 128, 768), (98304, 768, 1), 0); del buf181  # reuse
    buf198 = buf168; del buf168  # reuse
    buf199 = buf171; del buf171  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_39(c_void_p(buf199.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(sqrt_14.data_ptr()), c_void_p(sub_21.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    del primals_29
    del sqrt_14
    del sub_21
    buf200 = reinterpret_tensor(buf163, (512, 3072), (3072, 1), 0); del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (512, 768), (768, 1), 0), permute_297, out=buf200)
    del permute_297
    buf201 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (768, 512), (1, 768), 0), view_152, out=buf201)
    del view_152
    buf202 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf203 = reinterpret_tensor(buf200, (4, 128, 3072), (393216, 3072, 1), 0); del buf200  # reuse
    cpp_fused_gelu_gelu_backward_sum_40(c_void_p(buf203.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf202.data_ptr()))
    del addmm_40
    buf204 = reinterpret_tensor(buf196, (512, 768), (768, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (512, 3072), (3072, 1), 0), permute_301, out=buf204)
    del permute_301
    buf205 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (3072, 512), (1, 3072), 0), view_150, out=buf205)
    del view_150
    buf206 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf207 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf209 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf208 = buf198; del buf198  # reuse
    buf210 = buf195; del buf195  # reuse
    buf211 = buf199; del buf199  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_41(c_void_p(buf211.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(sqrt_13.data_ptr()), c_void_p(sub_20.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    del primals_27
    del sqrt_13
    del sub_20
    buf212 = buf204; del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), permute_305, out=buf212)
    del permute_305
    buf213 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (768, 512), (1, 768), 0), view_148, out=buf213)
    del view_148
    buf214 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf191, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf191  # reuse
    cpp_fused_clone_sum_42(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf212, (48, 128, 64), (8192, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_310, reinterpret_tensor(buf215, (48, 128, 64), (8192, 64, 1), 0), out=buf216)
    del permute_310
    buf217 = reinterpret_tensor(buf179, (48, 128, 128), (16384, 128, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf215, (48, 128, 64), (8192, 64, 1), 0), permute_311, out=buf217)
    del permute_311
    buf218 = buf178; del buf178  # reuse
    buf219 = reinterpret_tensor(buf217, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf217  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_43(c_void_p(buf219.data_ptr()), c_void_p(alias_52.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf218.data_ptr()))
    del alias_52
    buf220 = reinterpret_tensor(buf215, (48, 64, 128), (8192, 128, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_312, reinterpret_tensor(buf219, (48, 128, 128), (16384, 128, 1), 0), out=buf220)
    del permute_312
    buf221 = reinterpret_tensor(buf190, (48, 128, 64), (8192, 64, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (48, 128, 128), (16384, 128, 1), 0), permute_313, out=buf221)
    del permute_313
    buf222 = buf187; del buf187  # reuse
    cpp_fused_view_44(c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()))
    buf223 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf222, permute_316, out=buf223)
    del permute_316
    buf224 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (768, 512), (1, 768), 0), view_132, out=buf224)
    buf225 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf226 = buf183; del buf183  # reuse
    cpp_fused__unsafe_view_clone_sum_45(c_void_p(buf222.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = buf222; del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf226, permute_321, out=buf227)
    del permute_321
    buf228 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (768, 512), (1, 768), 0), view_132, out=buf228)
    buf229 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf230 = reinterpret_tensor(buf220, (512, 768), (768, 1), 0); del buf220  # reuse
    cpp_fused_sum_view_46(c_void_p(buf226.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = buf226; del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf230, permute_326, out=buf231)
    del permute_326
    buf232 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (768, 512), (1, 768), 0), view_132, out=buf232)
    del view_132
    buf233 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf234 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf237 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf235 = buf210; del buf210  # reuse
    buf236 = reinterpret_tensor(buf221, (4, 128, 768), (98304, 768, 1), 0); del buf221  # reuse
    buf238 = buf208; del buf208  # reuse
    buf239 = buf211; del buf211  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_47(c_void_p(buf239.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(sqrt_12.data_ptr()), c_void_p(sub_18.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del primals_25
    del sqrt_12
    del sub_18
    buf240 = reinterpret_tensor(buf203, (512, 3072), (3072, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (512, 768), (768, 1), 0), permute_330, out=buf240)
    del permute_330
    buf241 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (768, 512), (1, 768), 0), view_130, out=buf241)
    del view_130
    buf242 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf240, (4, 128, 3072), (393216, 3072, 1), 0); del buf240  # reuse
    cpp_fused_gelu_gelu_backward_sum_48(c_void_p(buf243.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf242.data_ptr()))
    del addmm_34
    buf244 = reinterpret_tensor(buf236, (512, 768), (768, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (512, 3072), (3072, 1), 0), permute_334, out=buf244)
    del permute_334
    buf245 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (3072, 512), (1, 3072), 0), view_128, out=buf245)
    del view_128
    buf246 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf247 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf249 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf248 = buf238; del buf238  # reuse
    buf250 = buf235; del buf235  # reuse
    buf251 = buf239; del buf239  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_49(c_void_p(buf251.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(sqrt_11.data_ptr()), c_void_p(sub_17.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    del primals_23
    del sqrt_11
    del sub_17
    buf252 = buf244; del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (512, 768), (768, 1), 0), permute_338, out=buf252)
    del permute_338
    buf253 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (768, 512), (1, 768), 0), view_126, out=buf253)
    del view_126
    buf254 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf255 = reinterpret_tensor(buf231, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf231  # reuse
    cpp_fused_clone_sum_50(c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    buf256 = reinterpret_tensor(buf252, (48, 128, 64), (8192, 64, 1), 0); del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_343, reinterpret_tensor(buf255, (48, 128, 64), (8192, 64, 1), 0), out=buf256)
    del permute_343
    buf257 = reinterpret_tensor(buf219, (48, 128, 128), (16384, 128, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf255, (48, 128, 64), (8192, 64, 1), 0), permute_344, out=buf257)
    del permute_344
    buf258 = buf218; del buf218  # reuse
    buf259 = reinterpret_tensor(buf257, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf257  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_51(c_void_p(buf259.data_ptr()), c_void_p(alias_55.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf258.data_ptr()))
    del alias_55
    buf260 = reinterpret_tensor(buf255, (48, 64, 128), (8192, 128, 1), 0); del buf255  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_345, reinterpret_tensor(buf259, (48, 128, 128), (16384, 128, 1), 0), out=buf260)
    del permute_345
    buf261 = reinterpret_tensor(buf230, (48, 128, 64), (8192, 64, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (48, 128, 128), (16384, 128, 1), 0), permute_346, out=buf261)
    del permute_346
    buf262 = buf227; del buf227  # reuse
    cpp_fused_view_52(c_void_p(buf256.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = reinterpret_tensor(buf256, (512, 768), (768, 1), 0); del buf256  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf262, permute_349, out=buf263)
    del permute_349
    buf264 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (768, 512), (1, 768), 0), view_110, out=buf264)
    buf265 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf266 = buf223; del buf223  # reuse
    cpp_fused__unsafe_view_clone_sum_53(c_void_p(buf262.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = buf262; del buf262  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf266, permute_354, out=buf267)
    del permute_354
    buf268 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (768, 512), (1, 768), 0), view_110, out=buf268)
    buf269 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf270 = reinterpret_tensor(buf260, (512, 768), (768, 1), 0); del buf260  # reuse
    cpp_fused_sum_view_54(c_void_p(buf266.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    buf271 = buf266; del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf270, permute_359, out=buf271)
    del permute_359
    buf272 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (768, 512), (1, 768), 0), view_110, out=buf272)
    del view_110
    buf273 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf274 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf277 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf275 = buf250; del buf250  # reuse
    buf276 = reinterpret_tensor(buf261, (4, 128, 768), (98304, 768, 1), 0); del buf261  # reuse
    buf278 = buf248; del buf248  # reuse
    buf279 = buf251; del buf251  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_55(c_void_p(buf279.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(sqrt_10.data_ptr()), c_void_p(sub_15.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()))
    del primals_21
    del sqrt_10
    del sub_15
    buf280 = reinterpret_tensor(buf243, (512, 3072), (3072, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (512, 768), (768, 1), 0), permute_363, out=buf280)
    del permute_363
    buf281 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (768, 512), (1, 768), 0), view_108, out=buf281)
    del view_108
    buf282 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf283 = reinterpret_tensor(buf280, (4, 128, 3072), (393216, 3072, 1), 0); del buf280  # reuse
    cpp_fused_gelu_gelu_backward_sum_56(c_void_p(buf283.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf282.data_ptr()))
    del addmm_28
    buf284 = reinterpret_tensor(buf276, (512, 768), (768, 1), 0); del buf276  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (512, 3072), (3072, 1), 0), permute_367, out=buf284)
    del permute_367
    buf285 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (3072, 512), (1, 3072), 0), view_106, out=buf285)
    del view_106
    buf286 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf287 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf289 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf288 = buf278; del buf278  # reuse
    buf290 = buf275; del buf275  # reuse
    buf291 = buf279; del buf279  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_57(c_void_p(buf291.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(sqrt_9.data_ptr()), c_void_p(sub_14.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    del primals_19
    del sqrt_9
    del sub_14
    buf292 = buf284; del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (512, 768), (768, 1), 0), permute_371, out=buf292)
    del permute_371
    buf293 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (768, 512), (1, 768), 0), view_104, out=buf293)
    del view_104
    buf294 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf295 = reinterpret_tensor(buf271, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf271  # reuse
    cpp_fused_clone_sum_58(c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    buf296 = reinterpret_tensor(buf292, (48, 128, 64), (8192, 64, 1), 0); del buf292  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_376, reinterpret_tensor(buf295, (48, 128, 64), (8192, 64, 1), 0), out=buf296)
    del permute_376
    buf297 = reinterpret_tensor(buf259, (48, 128, 128), (16384, 128, 1), 0); del buf259  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (48, 128, 64), (8192, 64, 1), 0), permute_377, out=buf297)
    del permute_377
    buf298 = buf258; del buf258  # reuse
    buf299 = reinterpret_tensor(buf297, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf297  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_59(c_void_p(buf299.data_ptr()), c_void_p(alias_58.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf298.data_ptr()))
    del alias_58
    buf300 = reinterpret_tensor(buf295, (48, 64, 128), (8192, 128, 1), 0); del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_378, reinterpret_tensor(buf299, (48, 128, 128), (16384, 128, 1), 0), out=buf300)
    del permute_378
    buf301 = reinterpret_tensor(buf270, (48, 128, 64), (8192, 64, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf299, (48, 128, 128), (16384, 128, 1), 0), permute_379, out=buf301)
    del permute_379
    buf302 = buf267; del buf267  # reuse
    cpp_fused_view_60(c_void_p(buf296.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf296, (512, 768), (768, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf302, permute_382, out=buf303)
    del permute_382
    buf304 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (768, 512), (1, 768), 0), view_88, out=buf304)
    buf305 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf306 = buf263; del buf263  # reuse
    cpp_fused__unsafe_view_clone_sum_61(c_void_p(buf302.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    buf307 = buf302; del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf306, permute_387, out=buf307)
    del permute_387
    buf308 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (768, 512), (1, 768), 0), view_88, out=buf308)
    buf309 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf310 = reinterpret_tensor(buf300, (512, 768), (768, 1), 0); del buf300  # reuse
    cpp_fused_sum_view_62(c_void_p(buf306.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    buf311 = buf306; del buf306  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf310, permute_392, out=buf311)
    del permute_392
    buf312 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (768, 512), (1, 768), 0), view_88, out=buf312)
    del view_88
    buf313 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf314 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf317 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf315 = buf290; del buf290  # reuse
    buf316 = reinterpret_tensor(buf301, (4, 128, 768), (98304, 768, 1), 0); del buf301  # reuse
    buf318 = buf288; del buf288  # reuse
    buf319 = buf291; del buf291  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_63(c_void_p(buf319.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(sqrt_8.data_ptr()), c_void_p(sub_12.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del primals_17
    del sqrt_8
    del sub_12
    buf320 = reinterpret_tensor(buf283, (512, 3072), (3072, 1), 0); del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (512, 768), (768, 1), 0), permute_396, out=buf320)
    del permute_396
    buf321 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (768, 512), (1, 768), 0), view_86, out=buf321)
    del view_86
    buf322 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf323 = reinterpret_tensor(buf320, (4, 128, 3072), (393216, 3072, 1), 0); del buf320  # reuse
    cpp_fused_gelu_gelu_backward_sum_64(c_void_p(buf323.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf322.data_ptr()))
    del addmm_22
    buf324 = reinterpret_tensor(buf316, (512, 768), (768, 1), 0); del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf323, (512, 3072), (3072, 1), 0), permute_400, out=buf324)
    del permute_400
    buf325 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf323, (3072, 512), (1, 3072), 0), view_84, out=buf325)
    del view_84
    buf326 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf327 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf329 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf328 = buf318; del buf318  # reuse
    buf330 = buf315; del buf315  # reuse
    buf331 = buf319; del buf319  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_65(c_void_p(buf331.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(sqrt_7.data_ptr()), c_void_p(sub_11.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()))
    del primals_15
    del sqrt_7
    del sub_11
    buf332 = buf324; del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (512, 768), (768, 1), 0), permute_404, out=buf332)
    del permute_404
    buf333 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (768, 512), (1, 768), 0), view_82, out=buf333)
    del view_82
    buf334 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf335 = reinterpret_tensor(buf311, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf311  # reuse
    cpp_fused_clone_sum_66(c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = reinterpret_tensor(buf332, (48, 128, 64), (8192, 64, 1), 0); del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_409, reinterpret_tensor(buf335, (48, 128, 64), (8192, 64, 1), 0), out=buf336)
    del permute_409
    buf337 = reinterpret_tensor(buf299, (48, 128, 128), (16384, 128, 1), 0); del buf299  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (48, 128, 64), (8192, 64, 1), 0), permute_410, out=buf337)
    del permute_410
    buf338 = buf298; del buf298  # reuse
    buf339 = reinterpret_tensor(buf337, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf337  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_67(c_void_p(buf339.data_ptr()), c_void_p(alias_61.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf338.data_ptr()))
    del alias_61
    buf340 = reinterpret_tensor(buf335, (48, 64, 128), (8192, 128, 1), 0); del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_411, reinterpret_tensor(buf339, (48, 128, 128), (16384, 128, 1), 0), out=buf340)
    del permute_411
    buf341 = reinterpret_tensor(buf310, (48, 128, 64), (8192, 64, 1), 0); del buf310  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (48, 128, 128), (16384, 128, 1), 0), permute_412, out=buf341)
    del permute_412
    buf342 = buf307; del buf307  # reuse
    cpp_fused_view_68(c_void_p(buf336.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = reinterpret_tensor(buf336, (512, 768), (768, 1), 0); del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, permute_415, out=buf343)
    del permute_415
    buf344 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (768, 512), (1, 768), 0), view_66, out=buf344)
    buf345 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf346 = buf303; del buf303  # reuse
    cpp_fused__unsafe_view_clone_sum_69(c_void_p(buf342.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    buf347 = buf342; del buf342  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf346, permute_420, out=buf347)
    del permute_420
    buf348 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf346, (768, 512), (1, 768), 0), view_66, out=buf348)
    buf349 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf350 = reinterpret_tensor(buf340, (512, 768), (768, 1), 0); del buf340  # reuse
    cpp_fused_sum_view_70(c_void_p(buf346.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = buf346; del buf346  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf350, permute_425, out=buf351)
    del permute_425
    buf352 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (768, 512), (1, 768), 0), view_66, out=buf352)
    del view_66
    buf353 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf354 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf357 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf355 = buf330; del buf330  # reuse
    buf356 = reinterpret_tensor(buf341, (4, 128, 768), (98304, 768, 1), 0); del buf341  # reuse
    buf358 = buf328; del buf328  # reuse
    buf359 = buf331; del buf331  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_71(c_void_p(buf359.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(sqrt_6.data_ptr()), c_void_p(sub_9.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()))
    del primals_13
    del sqrt_6
    del sub_9
    buf360 = reinterpret_tensor(buf323, (512, 3072), (3072, 1), 0); del buf323  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (512, 768), (768, 1), 0), permute_429, out=buf360)
    del permute_429
    buf361 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (768, 512), (1, 768), 0), view_64, out=buf361)
    del view_64
    buf362 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf360, (4, 128, 3072), (393216, 3072, 1), 0); del buf360  # reuse
    cpp_fused_gelu_gelu_backward_sum_72(c_void_p(buf363.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf362.data_ptr()))
    del addmm_16
    buf364 = reinterpret_tensor(buf356, (512, 768), (768, 1), 0); del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (512, 3072), (3072, 1), 0), permute_433, out=buf364)
    del permute_433
    buf365 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (3072, 512), (1, 3072), 0), view_62, out=buf365)
    del view_62
    buf366 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf367 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf369 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf368 = buf358; del buf358  # reuse
    buf370 = buf355; del buf355  # reuse
    buf371 = buf359; del buf359  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_73(c_void_p(buf371.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(sqrt_5.data_ptr()), c_void_p(sub_8.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()))
    del primals_11
    del sqrt_5
    del sub_8
    buf372 = buf364; del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (512, 768), (768, 1), 0), permute_437, out=buf372)
    del permute_437
    buf373 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (768, 512), (1, 768), 0), view_60, out=buf373)
    del view_60
    buf374 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf375 = reinterpret_tensor(buf351, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf351  # reuse
    cpp_fused_clone_sum_74(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = reinterpret_tensor(buf372, (48, 128, 64), (8192, 64, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_442, reinterpret_tensor(buf375, (48, 128, 64), (8192, 64, 1), 0), out=buf376)
    del permute_442
    buf377 = reinterpret_tensor(buf339, (48, 128, 128), (16384, 128, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (48, 128, 64), (8192, 64, 1), 0), permute_443, out=buf377)
    del permute_443
    buf378 = buf338; del buf338  # reuse
    buf379 = reinterpret_tensor(buf377, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf377  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_75(c_void_p(buf379.data_ptr()), c_void_p(alias_64.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf378.data_ptr()))
    del alias_64
    buf380 = reinterpret_tensor(buf375, (48, 64, 128), (8192, 128, 1), 0); del buf375  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_444, reinterpret_tensor(buf379, (48, 128, 128), (16384, 128, 1), 0), out=buf380)
    del permute_444
    buf381 = reinterpret_tensor(buf350, (48, 128, 64), (8192, 64, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (48, 128, 128), (16384, 128, 1), 0), permute_445, out=buf381)
    del permute_445
    buf382 = buf347; del buf347  # reuse
    cpp_fused_view_76(c_void_p(buf376.data_ptr()), c_void_p(buf382.data_ptr()))
    buf383 = reinterpret_tensor(buf376, (512, 768), (768, 1), 0); del buf376  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf382, permute_448, out=buf383)
    del permute_448
    buf384 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf382, (768, 512), (1, 768), 0), view_44, out=buf384)
    buf385 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf386 = buf343; del buf343  # reuse
    cpp_fused__unsafe_view_clone_sum_77(c_void_p(buf382.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = buf382; del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf386, permute_453, out=buf387)
    del permute_453
    buf388 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (768, 512), (1, 768), 0), view_44, out=buf388)
    buf389 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf390 = reinterpret_tensor(buf380, (512, 768), (768, 1), 0); del buf380  # reuse
    cpp_fused_sum_view_78(c_void_p(buf386.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = buf386; del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, permute_458, out=buf391)
    del permute_458
    buf392 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (768, 512), (1, 768), 0), view_44, out=buf392)
    del view_44
    buf393 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf394 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf397 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf395 = buf370; del buf370  # reuse
    buf396 = reinterpret_tensor(buf381, (4, 128, 768), (98304, 768, 1), 0); del buf381  # reuse
    buf398 = buf368; del buf368  # reuse
    buf399 = buf371; del buf371  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_79(c_void_p(buf399.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(sqrt_4.data_ptr()), c_void_p(sub_6.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()))
    del primals_9
    del sqrt_4
    del sub_6
    buf400 = reinterpret_tensor(buf363, (512, 3072), (3072, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (512, 768), (768, 1), 0), permute_462, out=buf400)
    del permute_462
    buf401 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (768, 512), (1, 768), 0), view_42, out=buf401)
    del view_42
    buf402 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf403 = reinterpret_tensor(buf400, (4, 128, 3072), (393216, 3072, 1), 0); del buf400  # reuse
    cpp_fused_gelu_gelu_backward_sum_80(c_void_p(buf403.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf402.data_ptr()))
    del addmm_10
    buf404 = reinterpret_tensor(buf396, (512, 768), (768, 1), 0); del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (512, 3072), (3072, 1), 0), permute_466, out=buf404)
    del permute_466
    buf405 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (3072, 512), (1, 3072), 0), view_40, out=buf405)
    del view_40
    buf406 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf407 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf409 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf408 = buf398; del buf398  # reuse
    buf410 = buf395; del buf395  # reuse
    buf411 = buf399; del buf399  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_81(c_void_p(buf411.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(sqrt_3.data_ptr()), c_void_p(sub_5.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()))
    del primals_7
    del sqrt_3
    del sub_5
    buf412 = buf404; del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (512, 768), (768, 1), 0), permute_470, out=buf412)
    del permute_470
    buf413 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (768, 512), (1, 768), 0), view_38, out=buf413)
    del view_38
    buf414 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf415 = reinterpret_tensor(buf391, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf391  # reuse
    cpp_fused_clone_sum_82(c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    buf416 = reinterpret_tensor(buf412, (48, 128, 64), (8192, 64, 1), 0); del buf412  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_475, reinterpret_tensor(buf415, (48, 128, 64), (8192, 64, 1), 0), out=buf416)
    del permute_475
    buf417 = reinterpret_tensor(buf379, (48, 128, 128), (16384, 128, 1), 0); del buf379  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf415, (48, 128, 64), (8192, 64, 1), 0), permute_476, out=buf417)
    del permute_476
    buf418 = buf378; del buf378  # reuse
    buf419 = reinterpret_tensor(buf417, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf417  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_83(c_void_p(buf419.data_ptr()), c_void_p(alias_67.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf418.data_ptr()))
    del alias_67
    buf420 = reinterpret_tensor(buf415, (48, 64, 128), (8192, 128, 1), 0); del buf415  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_477, reinterpret_tensor(buf419, (48, 128, 128), (16384, 128, 1), 0), out=buf420)
    del permute_477
    buf421 = reinterpret_tensor(buf390, (48, 128, 64), (8192, 64, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf419, (48, 128, 128), (16384, 128, 1), 0), permute_478, out=buf421)
    del permute_478
    buf422 = buf387; del buf387  # reuse
    cpp_fused_view_84(c_void_p(buf416.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf416, (512, 768), (768, 1), 0); del buf416  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, permute_481, out=buf423)
    del permute_481
    buf424 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_22, out=buf424)
    buf425 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf426 = buf383; del buf383  # reuse
    cpp_fused__unsafe_view_clone_sum_85(c_void_p(buf422.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    buf427 = buf422; del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf426, permute_486, out=buf427)
    del permute_486
    buf428 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (768, 512), (1, 768), 0), view_22, out=buf428)
    buf429 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf430 = reinterpret_tensor(buf420, (512, 768), (768, 1), 0); del buf420  # reuse
    cpp_fused_sum_view_86(c_void_p(buf426.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    buf431 = buf426; del buf426  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf430, permute_491, out=buf431)
    del permute_491
    buf432 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (768, 512), (1, 768), 0), view_22, out=buf432)
    del view_22
    buf433 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf434 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf437 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf435 = buf410; del buf410  # reuse
    buf436 = reinterpret_tensor(buf421, (4, 128, 768), (98304, 768, 1), 0); del buf421  # reuse
    buf438 = buf408; del buf408  # reuse
    buf439 = buf411; del buf411  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_87(c_void_p(buf439.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(sqrt_2.data_ptr()), c_void_p(sub_3.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()))
    del primals_5
    del sqrt_2
    del sub_3
    buf440 = reinterpret_tensor(buf403, (512, 3072), (3072, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf439, (512, 768), (768, 1), 0), permute_495, out=buf440)
    del permute_495
    buf441 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf439, (768, 512), (1, 768), 0), view_20, out=buf441)
    del view_20
    buf442 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf443 = reinterpret_tensor(buf440, (4, 128, 3072), (393216, 3072, 1), 0); del buf440  # reuse
    cpp_fused_gelu_gelu_backward_sum_88(c_void_p(buf443.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf442.data_ptr()))
    del addmm_4
    buf444 = reinterpret_tensor(buf436, (512, 768), (768, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf443, (512, 3072), (3072, 1), 0), permute_499, out=buf444)
    del permute_499
    buf445 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf443, (3072, 512), (1, 3072), 0), view_18, out=buf445)
    del view_18
    buf446 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf447 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf449 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf448 = buf438; del buf438  # reuse
    buf450 = buf435; del buf435  # reuse
    buf451 = buf439; del buf439  # reuse
    cpp_fused_add_div_eq_masked_fill_mul_neg_sum_89(c_void_p(buf451.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(sqrt_1.data_ptr()), c_void_p(sub_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf450.data_ptr()))
    del buf443
    del primals_3
    del sqrt_1
    del sub_2
    buf452 = buf444; del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf451, (512, 768), (768, 1), 0), permute_503, out=buf452)
    del permute_503
    buf453 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf451, (768, 512), (1, 768), 0), view_16, out=buf453)
    del view_16
    buf454 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf455 = reinterpret_tensor(buf431, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf431  # reuse
    cpp_fused_clone_sum_90(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()))
    buf456 = reinterpret_tensor(buf452, (48, 128, 64), (8192, 64, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_508, reinterpret_tensor(buf455, (48, 128, 64), (8192, 64, 1), 0), out=buf456)
    del permute_508
    buf457 = reinterpret_tensor(buf419, (48, 128, 128), (16384, 128, 1), 0); del buf419  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf455, (48, 128, 64), (8192, 64, 1), 0), permute_509, out=buf457)
    del permute_509
    buf458 = buf418; del buf418  # reuse
    buf459 = reinterpret_tensor(buf457, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf457  # reuse
    cpp_fused__softmax_backward_data_div_eq_masked_fill_91(c_void_p(buf459.data_ptr()), c_void_p(alias_70.data_ptr()), c_void_p(unsqueeze_1.data_ptr()), c_void_p(buf458.data_ptr()))
    del alias_70
    del buf458
    del unsqueeze_1
    buf460 = reinterpret_tensor(buf455, (48, 64, 128), (8192, 128, 1), 0); del buf455  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_510, reinterpret_tensor(buf459, (48, 128, 128), (16384, 128, 1), 0), out=buf460)
    del permute_510
    buf461 = reinterpret_tensor(buf430, (48, 128, 64), (8192, 64, 1), 0); del buf430  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf459, (48, 128, 128), (16384, 128, 1), 0), permute_511, out=buf461)
    del buf459
    del permute_511
    buf462 = buf427; del buf427  # reuse
    cpp_fused_view_92(c_void_p(buf456.data_ptr()), c_void_p(buf462.data_ptr()))
    buf463 = reinterpret_tensor(buf456, (512, 768), (768, 1), 0); del buf456  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf462, permute_514, out=buf463)
    del permute_514
    buf464 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (768, 512), (1, 768), 0), view, out=buf464)
    buf465 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf466 = buf423; del buf423  # reuse
    cpp_fused__unsafe_view_clone_sum_93(c_void_p(buf462.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    buf467 = buf462; del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf466, permute_519, out=buf467)
    del permute_519
    buf468 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (768, 512), (1, 768), 0), view, out=buf468)
    buf469 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf470 = reinterpret_tensor(buf460, (512, 768), (768, 1), 0); del buf460  # reuse
    cpp_fused_sum_view_94(c_void_p(buf466.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    buf471 = buf466; del buf466  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf470, permute_524, out=buf471)
    del permute_524
    buf472 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (768, 512), (1, 768), 0), view, out=buf472)
    del view
    buf473 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf474 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf477 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf475 = buf450; del buf450  # reuse
    buf476 = reinterpret_tensor(buf461, (4, 128, 768), (98304, 768, 1), 0); del buf461  # reuse
    buf478 = buf448; del buf448  # reuse
    buf479 = buf451; del buf451  # reuse
    buf481 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf485 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf480 = empty((3, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_eq_masked_fill_mul_neg_sum_95(c_void_p(buf479.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(sqrt.data_ptr()), c_void_p(sub.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf480.data_ptr()))
    del buf463
    del buf467
    del buf470
    del buf471
    del buf475
    del buf476
    del buf478
    del buf479
    del primals_1
    del sqrt
    del sub
    aten.index_put_(buf480, [primals_197], buf481, True)
    del buf481
    del primals_197
    buf484 = empty((20005, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_96(c_void_p(buf484.data_ptr()))
    aten.index_put_(buf484, [primals_196], buf485, True)
    del buf485
    del primals_196
    return (reinterpret_tensor(buf477, (768, ), (1, ), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), reinterpret_tensor(buf449, (768, ), (1, ), 0), reinterpret_tensor(buf447, (768, ), (1, ), 0), reinterpret_tensor(buf437, (768, ), (1, ), 0), reinterpret_tensor(buf434, (768, ), (1, ), 0), reinterpret_tensor(buf409, (768, ), (1, ), 0), reinterpret_tensor(buf407, (768, ), (1, ), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), reinterpret_tensor(buf369, (768, ), (1, ), 0), reinterpret_tensor(buf367, (768, ), (1, ), 0), reinterpret_tensor(buf357, (768, ), (1, ), 0), reinterpret_tensor(buf354, (768, ), (1, ), 0), reinterpret_tensor(buf329, (768, ), (1, ), 0), reinterpret_tensor(buf327, (768, ), (1, ), 0), reinterpret_tensor(buf317, (768, ), (1, ), 0), reinterpret_tensor(buf314, (768, ), (1, ), 0), reinterpret_tensor(buf289, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, ), (1, ), 0), reinterpret_tensor(buf277, (768, ), (1, ), 0), reinterpret_tensor(buf274, (768, ), (1, ), 0), reinterpret_tensor(buf249, (768, ), (1, ), 0), reinterpret_tensor(buf247, (768, ), (1, ), 0), reinterpret_tensor(buf237, (768, ), (1, ), 0), reinterpret_tensor(buf234, (768, ), (1, ), 0), reinterpret_tensor(buf209, (768, ), (1, ), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, ), (1, ), 0), reinterpret_tensor(buf194, (768, ), (1, ), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, ), (1, ), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), reinterpret_tensor(buf154, (768, ), (1, ), 0), reinterpret_tensor(buf129, (768, ), (1, ), 0), reinterpret_tensor(buf127, (768, ), (1, ), 0), reinterpret_tensor(buf117, (768, ), (1, ), 0), reinterpret_tensor(buf114, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf77, (768, ), (1, ), 0), reinterpret_tensor(buf74, (768, ), (1, ), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf37, (768, ), (1, ), 0), reinterpret_tensor(buf34, (768, ), (1, ), 0), reinterpret_tensor(buf9, (768, ), (1, ), 0), reinterpret_tensor(buf7, (768, ), (1, ), 0), buf484, buf480, reinterpret_tensor(buf472, (768, 768), (768, 1), 0), reinterpret_tensor(buf473, (768, ), (1, ), 0), reinterpret_tensor(buf468, (768, 768), (768, 1), 0), reinterpret_tensor(buf469, (768, ), (1, ), 0), reinterpret_tensor(buf464, (768, 768), (768, 1), 0), reinterpret_tensor(buf465, (768, ), (1, ), 0), reinterpret_tensor(buf453, (768, 768), (768, 1), 0), reinterpret_tensor(buf454, (768, ), (1, ), 0), reinterpret_tensor(buf445, (3072, 768), (768, 1), 0), reinterpret_tensor(buf446, (3072, ), (1, ), 0), reinterpret_tensor(buf441, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf442, (768, ), (1, ), 0), reinterpret_tensor(buf432, (768, 768), (768, 1), 0), reinterpret_tensor(buf433, (768, ), (1, ), 0), reinterpret_tensor(buf428, (768, 768), (768, 1), 0), reinterpret_tensor(buf429, (768, ), (1, ), 0), reinterpret_tensor(buf424, (768, 768), (768, 1), 0), reinterpret_tensor(buf425, (768, ), (1, ), 0), reinterpret_tensor(buf413, (768, 768), (768, 1), 0), reinterpret_tensor(buf414, (768, ), (1, ), 0), reinterpret_tensor(buf405, (3072, 768), (768, 1), 0), reinterpret_tensor(buf406, (3072, ), (1, ), 0), reinterpret_tensor(buf401, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf402, (768, ), (1, ), 0), reinterpret_tensor(buf392, (768, 768), (768, 1), 0), reinterpret_tensor(buf393, (768, ), (1, ), 0), reinterpret_tensor(buf388, (768, 768), (768, 1), 0), reinterpret_tensor(buf389, (768, ), (1, ), 0), reinterpret_tensor(buf384, (768, 768), (768, 1), 0), reinterpret_tensor(buf385, (768, ), (1, ), 0), reinterpret_tensor(buf373, (768, 768), (768, 1), 0), reinterpret_tensor(buf374, (768, ), (1, ), 0), reinterpret_tensor(buf365, (3072, 768), (768, 1), 0), reinterpret_tensor(buf366, (3072, ), (1, ), 0), reinterpret_tensor(buf361, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf362, (768, ), (1, ), 0), reinterpret_tensor(buf352, (768, 768), (768, 1), 0), reinterpret_tensor(buf353, (768, ), (1, ), 0), reinterpret_tensor(buf348, (768, 768), (768, 1), 0), reinterpret_tensor(buf349, (768, ), (1, ), 0), reinterpret_tensor(buf344, (768, 768), (768, 1), 0), reinterpret_tensor(buf345, (768, ), (1, ), 0), reinterpret_tensor(buf333, (768, 768), (768, 1), 0), reinterpret_tensor(buf334, (768, ), (1, ), 0), reinterpret_tensor(buf325, (3072, 768), (768, 1), 0), reinterpret_tensor(buf326, (3072, ), (1, ), 0), reinterpret_tensor(buf321, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf322, (768, ), (1, ), 0), reinterpret_tensor(buf312, (768, 768), (768, 1), 0), reinterpret_tensor(buf313, (768, ), (1, ), 0), reinterpret_tensor(buf308, (768, 768), (768, 1), 0), reinterpret_tensor(buf309, (768, ), (1, ), 0), reinterpret_tensor(buf304, (768, 768), (768, 1), 0), reinterpret_tensor(buf305, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, 768), (768, 1), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), reinterpret_tensor(buf285, (3072, 768), (768, 1), 0), reinterpret_tensor(buf286, (3072, ), (1, ), 0), reinterpret_tensor(buf281, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf282, (768, ), (1, ), 0), reinterpret_tensor(buf272, (768, 768), (768, 1), 0), reinterpret_tensor(buf273, (768, ), (1, ), 0), reinterpret_tensor(buf268, (768, 768), (768, 1), 0), reinterpret_tensor(buf269, (768, ), (1, ), 0), reinterpret_tensor(buf264, (768, 768), (768, 1), 0), reinterpret_tensor(buf265, (768, ), (1, ), 0), reinterpret_tensor(buf253, (768, 768), (768, 1), 0), reinterpret_tensor(buf254, (768, ), (1, ), 0), reinterpret_tensor(buf245, (3072, 768), (768, 1), 0), reinterpret_tensor(buf246, (3072, ), (1, ), 0), reinterpret_tensor(buf241, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf242, (768, ), (1, ), 0), reinterpret_tensor(buf232, (768, 768), (768, 1), 0), reinterpret_tensor(buf233, (768, ), (1, ), 0), reinterpret_tensor(buf228, (768, 768), (768, 1), 0), reinterpret_tensor(buf229, (768, ), (1, ), 0), reinterpret_tensor(buf224, (768, 768), (768, 1), 0), reinterpret_tensor(buf225, (768, ), (1, ), 0), reinterpret_tensor(buf213, (768, 768), (768, 1), 0), reinterpret_tensor(buf214, (768, ), (1, ), 0), reinterpret_tensor(buf205, (3072, 768), (768, 1), 0), reinterpret_tensor(buf206, (3072, ), (1, ), 0), reinterpret_tensor(buf201, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf202, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, 768), (768, 1), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), reinterpret_tensor(buf188, (768, 768), (768, 1), 0), reinterpret_tensor(buf189, (768, ), (1, ), 0), reinterpret_tensor(buf184, (768, 768), (768, 1), 0), reinterpret_tensor(buf185, (768, ), (1, ), 0), reinterpret_tensor(buf173, (768, 768), (768, 1), 0), reinterpret_tensor(buf174, (768, ), (1, ), 0), reinterpret_tensor(buf165, (3072, 768), (768, 1), 0), reinterpret_tensor(buf166, (3072, ), (1, ), 0), reinterpret_tensor(buf161, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), reinterpret_tensor(buf152, (768, 768), (768, 1), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), reinterpret_tensor(buf148, (768, 768), (768, 1), 0), reinterpret_tensor(buf149, (768, ), (1, ), 0), reinterpret_tensor(buf144, (768, 768), (768, 1), 0), reinterpret_tensor(buf145, (768, ), (1, ), 0), reinterpret_tensor(buf133, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), reinterpret_tensor(buf125, (3072, 768), (768, 1), 0), reinterpret_tensor(buf126, (3072, ), (1, ), 0), reinterpret_tensor(buf121, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf122, (768, ), (1, ), 0), reinterpret_tensor(buf112, (768, 768), (768, 1), 0), reinterpret_tensor(buf113, (768, ), (1, ), 0), reinterpret_tensor(buf108, (768, 768), (768, 1), 0), reinterpret_tensor(buf109, (768, ), (1, ), 0), reinterpret_tensor(buf104, (768, 768), (768, 1), 0), reinterpret_tensor(buf105, (768, ), (1, ), 0), reinterpret_tensor(buf93, (768, 768), (768, 1), 0), reinterpret_tensor(buf94, (768, ), (1, ), 0), reinterpret_tensor(buf85, (3072, 768), (768, 1), 0), reinterpret_tensor(buf86, (3072, ), (1, ), 0), reinterpret_tensor(buf81, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf82, (768, ), (1, ), 0), reinterpret_tensor(buf72, (768, 768), (768, 1), 0), reinterpret_tensor(buf73, (768, ), (1, ), 0), reinterpret_tensor(buf68, (768, 768), (768, 1), 0), reinterpret_tensor(buf69, (768, ), (1, ), 0), reinterpret_tensor(buf64, (768, 768), (768, 1), 0), reinterpret_tensor(buf65, (768, ), (1, ), 0), reinterpret_tensor(buf53, (768, 768), (768, 1), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), reinterpret_tensor(buf45, (3072, 768), (768, 1), 0), reinterpret_tensor(buf46, (3072, ), (1, ), 0), reinterpret_tensor(buf41, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf32, (768, 768), (768, 1), 0), reinterpret_tensor(buf33, (768, ), (1, ), 0), reinterpret_tensor(buf28, (768, 768), (768, 1), 0), reinterpret_tensor(buf29, (768, ), (1, ), 0), reinterpret_tensor(buf24, (768, 768), (768, 1), 0), reinterpret_tensor(buf25, (768, ), (1, ), 0), reinterpret_tensor(buf13, (768, 768), (768, 1), 0), reinterpret_tensor(buf14, (768, ), (1, ), 0), reinterpret_tensor(buf5, (3072, 768), (768, 1), 0), reinterpret_tensor(buf6, (3072, ), (1, ), 0), reinterpret_tensor(buf1, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf2, (768, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_197 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    unsqueeze_1 = rand_strided((4, 1, 128, 128), (16384, 16384, 128, 1), device='cpu', dtype=torch.bool)
    sqrt = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_1 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_2 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_2 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_3 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_3 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_5 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_4 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_6 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_5 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_8 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_6 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_9 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_7 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_11 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_8 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_12 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_9 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_14 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_10 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_15 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_11 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_17 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_12 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_18 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_13 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_20 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_14 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_21 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_15 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_23 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_16 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_24 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_17 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_26 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_18 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_27 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_19 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_29 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_20 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_30 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_21 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_32 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    sqrt_22 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_33 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sqrt_23 = rand_strided((4, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    sub_35 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_148 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_173 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_40 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_181 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_202 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_46 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_246 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_247 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_272 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_49 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_279 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_293 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_297 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_301 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_305 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_52 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_312 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_321 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_326 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_330 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_55 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_345 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_346 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_359 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_363 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_367 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_371 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_58 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_378 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_379 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_387 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_392 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_396 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_400 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_61 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_411 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_412 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_425 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_429 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_433 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_437 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_442 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_64 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_444 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_445 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_458 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_462 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_466 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_470 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_67 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_477 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_478 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_486 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_491 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_495 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_499 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_503 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_508 = rand_strided((48, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_70 = rand_strided((4, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_510 = rand_strided((48, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_511 = rand_strided((48, 128, 64), (8192, 1, 128), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_519 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_524 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, unsqueeze_1, sqrt, sub, view, view_16, sqrt_1, sub_2, view_18, addmm_4, view_20, sqrt_2, sub_3, view_22, view_38, sqrt_3, sub_5, view_40, addmm_10, view_42, sqrt_4, sub_6, view_44, view_60, sqrt_5, sub_8, view_62, addmm_16, view_64, sqrt_6, sub_9, view_66, view_82, sqrt_7, sub_11, view_84, addmm_22, view_86, sqrt_8, sub_12, view_88, view_104, sqrt_9, sub_14, view_106, addmm_28, view_108, sqrt_10, sub_15, view_110, view_126, sqrt_11, sub_17, view_128, addmm_34, view_130, sqrt_12, sub_18, view_132, view_148, sqrt_13, sub_20, view_150, addmm_40, view_152, sqrt_14, sub_21, view_154, view_170, sqrt_15, sub_23, view_172, addmm_46, view_174, sqrt_16, sub_24, view_176, view_192, sqrt_17, sub_26, view_194, addmm_52, view_196, sqrt_18, sub_27, view_198, view_214, sqrt_19, sub_29, view_216, addmm_58, view_218, sqrt_20, sub_30, view_220, view_236, sqrt_21, sub_32, view_238, addmm_64, view_240, sqrt_22, sub_33, view_242, view_258, sqrt_23, sub_35, view_260, addmm_70, view_262, permute_132, permute_136, permute_140, permute_145, permute_146, alias_37, permute_147, permute_148, permute_151, permute_156, permute_161, permute_165, permute_169, permute_173, permute_178, permute_179, alias_40, permute_180, permute_181, permute_184, permute_189, permute_194, permute_198, permute_202, permute_206, permute_211, permute_212, alias_43, permute_213, permute_214, permute_217, permute_222, permute_227, permute_231, permute_235, permute_239, permute_244, permute_245, alias_46, permute_246, permute_247, permute_250, permute_255, permute_260, permute_264, permute_268, permute_272, permute_277, permute_278, alias_49, permute_279, permute_280, permute_283, permute_288, permute_293, permute_297, permute_301, permute_305, permute_310, permute_311, alias_52, permute_312, permute_313, permute_316, permute_321, permute_326, permute_330, permute_334, permute_338, permute_343, permute_344, alias_55, permute_345, permute_346, permute_349, permute_354, permute_359, permute_363, permute_367, permute_371, permute_376, permute_377, alias_58, permute_378, permute_379, permute_382, permute_387, permute_392, permute_396, permute_400, permute_404, permute_409, permute_410, alias_61, permute_411, permute_412, permute_415, permute_420, permute_425, permute_429, permute_433, permute_437, permute_442, permute_443, alias_64, permute_444, permute_445, permute_448, permute_453, permute_458, permute_462, permute_466, permute_470, permute_475, permute_476, alias_67, permute_477, permute_478, permute_481, permute_486, permute_491, permute_495, permute_499, permute_503, permute_508, permute_509, alias_70, permute_510, permute_511, permute_514, permute_519, permute_524, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
