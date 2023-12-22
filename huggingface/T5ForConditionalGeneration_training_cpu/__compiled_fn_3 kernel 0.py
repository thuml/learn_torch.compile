
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


cpp_fused_embedding_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1 + ((-1L)*x0));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = c10::convert<long>(tmp2);
                    auto tmp4 = static_cast<long>(16);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 + tmp1);
                    auto tmp7 = std::abs(tmp0);
                    auto tmp8 = static_cast<long>(8);
                    auto tmp9 = tmp7 < tmp8;
                    auto tmp10 = c10::convert<float>(tmp7);
                    auto tmp11 = static_cast<float>(8.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::log(tmp12);
                    auto tmp14 = static_cast<float>(2.772588722239781);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp11);
                    auto tmp17 = c10::convert<long>(tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 + tmp8);
                    auto tmp19 = static_cast<long>(15);
                    auto tmp20 = min_propagate_nan(tmp18, tmp19);
                    auto tmp21 = tmp9 ? tmp7 : tmp20;
                    auto tmp22 = decltype(tmp6)(tmp6 + tmp21);
                    out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr1[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<long>(tmp3);
                            auto tmp5 = static_cast<long>(16);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                            auto tmp8 = std::abs(tmp1);
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = c10::convert<float>(tmp8);
                            auto tmp12 = static_cast<float>(8.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = std::log(tmp13);
                            auto tmp15 = static_cast<float>(2.772588722239781);
                            auto tmp16 = tmp14 / tmp15;
                            auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                            auto tmp20 = static_cast<long>(15);
                            auto tmp21 = min_propagate_nan(tmp19, tmp20);
                            auto tmp22 = tmp10 ? tmp8 : tmp21;
                            auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                            auto tmp24 = decltype(tmp23)(tmp23 + 32);
                            auto tmp25 = tmp23 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp23;
                            TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                            auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                            auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp29 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>(x2 + ((-1L)*x1));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 > tmp2;
                        auto tmp4 = c10::convert<long>(tmp3);
                        auto tmp5 = static_cast<long>(16);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp6)(tmp6 + tmp2);
                        auto tmp8 = std::abs(tmp1);
                        auto tmp9 = static_cast<long>(8);
                        auto tmp10 = tmp8 < tmp9;
                        auto tmp11 = c10::convert<float>(tmp8);
                        auto tmp12 = static_cast<float>(8.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = std::log(tmp13);
                        auto tmp15 = static_cast<float>(2.772588722239781);
                        auto tmp16 = tmp14 / tmp15;
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                        auto tmp18 = c10::convert<long>(tmp17);
                        auto tmp19 = decltype(tmp18)(tmp18 + tmp9);
                        auto tmp20 = static_cast<long>(15);
                        auto tmp21 = min_propagate_nan(tmp19, tmp20);
                        auto tmp22 = tmp10 ? tmp8 : tmp21;
                        auto tmp23 = decltype(tmp7)(tmp7 + tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + 32);
                        auto tmp25 = tmp23 < 0;
                        auto tmp26 = tmp25 ? tmp24 : tmp23;
                        TORCH_CHECK((0 <= tmp26) & (tmp26 < 32L), "index out of bounds: 0 <= tmp26 < 32L")
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (8L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = std::exp(tmp30);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_embedding_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>((-1L)*(std::min(0L, x1 + ((-1L)*x0))));
                    auto tmp1 = static_cast<long>(16);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = c10::convert<float>(tmp0);
                    auto tmp4 = static_cast<float>(16.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = std::log(tmp5);
                    auto tmp7 = static_cast<float>(2.0794415416798357);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp4);
                    auto tmp10 = c10::convert<long>(tmp9);
                    auto tmp11 = decltype(tmp10)(tmp10 + tmp1);
                    auto tmp12 = static_cast<long>(31);
                    auto tmp13 = min_propagate_nan(tmp11, tmp12);
                    auto tmp14 = tmp2 ? tmp0 : tmp13;
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr1[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                            auto tmp2 = static_cast<long>(16);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = c10::convert<float>(tmp1);
                            auto tmp5 = static_cast<float>(16.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = std::log(tmp6);
                            auto tmp8 = static_cast<float>(2.0794415416798357);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                            auto tmp11 = c10::convert<long>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                            auto tmp13 = static_cast<long>(31);
                            auto tmp14 = min_propagate_nan(tmp12, tmp13);
                            auto tmp15 = tmp3 ? tmp1 : tmp14;
                            auto tmp16 = static_cast<long>(0);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = decltype(tmp17)(tmp17 + 32);
                            auto tmp19 = tmp17 < 0;
                            auto tmp20 = tmp19 ? tmp18 : tmp17;
                            TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                            auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x2);
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp31);
                        }
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp32 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x2 + ((-1L)*x1))));
                        auto tmp2 = static_cast<long>(16);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = c10::convert<float>(tmp1);
                        auto tmp5 = static_cast<float>(16.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = std::log(tmp6);
                        auto tmp8 = static_cast<float>(2.0794415416798357);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp5);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                        auto tmp13 = static_cast<long>(31);
                        auto tmp14 = min_propagate_nan(tmp12, tmp13);
                        auto tmp15 = tmp3 ? tmp1 : tmp14;
                        auto tmp16 = static_cast<long>(0);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = decltype(tmp17)(tmp17 + 32);
                        auto tmp19 = tmp17 < 0;
                        auto tmp20 = tmp19 ? tmp18 : tmp17;
                        TORCH_CHECK((0 <= tmp20) & (tmp20 < 32L), "index out of bounds: 0 <= tmp20 < 32L")
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (8L*tmp20))];
                        auto tmp22 = c10::convert<long>(x2);
                        auto tmp23 = c10::convert<long>(x1);
                        auto tmp24 = tmp22 <= tmp23;
                        auto tmp25 = c10::convert<float>(tmp24);
                        auto tmp26 = static_cast<float>(1.0);
                        auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                        auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                        auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                in_out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.04419417382415922);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (32128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 32128);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 32128L), "index out of bounds: 0 <= tmp8 < 32128L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (32128L*x0))];
                        auto tmp10 = decltype(tmp9)(-tmp9);
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = tmp2 ? tmp10 : tmp11;
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp12;
                    }
                    out_ptr3[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr4[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = out_ptr4[static_cast<long>(0L)];
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp3 = tmp2 / tmp1;
                out_ptr5[static_cast<long>(0L)] = tmp1;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (32128, 512), (512, 1))
    assert_size_stride(primals_34, (512, 512), (512, 1))
    assert_size_stride(primals_35, (512, 512), (512, 1))
    assert_size_stride(primals_36, (512, 512), (512, 1))
    assert_size_stride(primals_37, (32, 8), (8, 1))
    assert_size_stride(primals_38, (512, 512), (512, 1))
    assert_size_stride(primals_39, (2048, 512), (512, 1))
    assert_size_stride(primals_40, (512, 2048), (2048, 1))
    assert_size_stride(primals_41, (512, 512), (512, 1))
    assert_size_stride(primals_42, (512, 512), (512, 1))
    assert_size_stride(primals_43, (512, 512), (512, 1))
    assert_size_stride(primals_44, (512, 512), (512, 1))
    assert_size_stride(primals_45, (2048, 512), (512, 1))
    assert_size_stride(primals_46, (512, 2048), (2048, 1))
    assert_size_stride(primals_47, (512, 512), (512, 1))
    assert_size_stride(primals_48, (512, 512), (512, 1))
    assert_size_stride(primals_49, (512, 512), (512, 1))
    assert_size_stride(primals_50, (512, 512), (512, 1))
    assert_size_stride(primals_51, (2048, 512), (512, 1))
    assert_size_stride(primals_52, (512, 2048), (2048, 1))
    assert_size_stride(primals_53, (512, 512), (512, 1))
    assert_size_stride(primals_54, (512, 512), (512, 1))
    assert_size_stride(primals_55, (512, 512), (512, 1))
    assert_size_stride(primals_56, (512, 512), (512, 1))
    assert_size_stride(primals_57, (2048, 512), (512, 1))
    assert_size_stride(primals_58, (512, 2048), (2048, 1))
    assert_size_stride(primals_59, (512, 512), (512, 1))
    assert_size_stride(primals_60, (512, 512), (512, 1))
    assert_size_stride(primals_61, (512, 512), (512, 1))
    assert_size_stride(primals_62, (512, 512), (512, 1))
    assert_size_stride(primals_63, (2048, 512), (512, 1))
    assert_size_stride(primals_64, (512, 2048), (2048, 1))
    assert_size_stride(primals_65, (512, 512), (512, 1))
    assert_size_stride(primals_66, (512, 512), (512, 1))
    assert_size_stride(primals_67, (512, 512), (512, 1))
    assert_size_stride(primals_68, (512, 512), (512, 1))
    assert_size_stride(primals_69, (2048, 512), (512, 1))
    assert_size_stride(primals_70, (512, 2048), (2048, 1))
    assert_size_stride(primals_71, (512, 512), (512, 1))
    assert_size_stride(primals_72, (512, 512), (512, 1))
    assert_size_stride(primals_73, (512, 512), (512, 1))
    assert_size_stride(primals_74, (32, 8), (8, 1))
    assert_size_stride(primals_75, (512, 512), (512, 1))
    assert_size_stride(primals_76, (512, 512), (512, 1))
    assert_size_stride(primals_77, (512, 512), (512, 1))
    assert_size_stride(primals_78, (512, 512), (512, 1))
    assert_size_stride(primals_79, (512, 512), (512, 1))
    assert_size_stride(primals_80, (2048, 512), (512, 1))
    assert_size_stride(primals_81, (512, 2048), (2048, 1))
    assert_size_stride(primals_82, (512, 512), (512, 1))
    assert_size_stride(primals_83, (512, 512), (512, 1))
    assert_size_stride(primals_84, (512, 512), (512, 1))
    assert_size_stride(primals_85, (512, 512), (512, 1))
    assert_size_stride(primals_86, (512, 512), (512, 1))
    assert_size_stride(primals_87, (512, 512), (512, 1))
    assert_size_stride(primals_88, (512, 512), (512, 1))
    assert_size_stride(primals_89, (512, 512), (512, 1))
    assert_size_stride(primals_90, (2048, 512), (512, 1))
    assert_size_stride(primals_91, (512, 2048), (2048, 1))
    assert_size_stride(primals_92, (512, 512), (512, 1))
    assert_size_stride(primals_93, (512, 512), (512, 1))
    assert_size_stride(primals_94, (512, 512), (512, 1))
    assert_size_stride(primals_95, (512, 512), (512, 1))
    assert_size_stride(primals_96, (512, 512), (512, 1))
    assert_size_stride(primals_97, (512, 512), (512, 1))
    assert_size_stride(primals_98, (512, 512), (512, 1))
    assert_size_stride(primals_99, (512, 512), (512, 1))
    assert_size_stride(primals_100, (2048, 512), (512, 1))
    assert_size_stride(primals_101, (512, 2048), (2048, 1))
    assert_size_stride(primals_102, (512, 512), (512, 1))
    assert_size_stride(primals_103, (512, 512), (512, 1))
    assert_size_stride(primals_104, (512, 512), (512, 1))
    assert_size_stride(primals_105, (512, 512), (512, 1))
    assert_size_stride(primals_106, (512, 512), (512, 1))
    assert_size_stride(primals_107, (512, 512), (512, 1))
    assert_size_stride(primals_108, (512, 512), (512, 1))
    assert_size_stride(primals_109, (512, 512), (512, 1))
    assert_size_stride(primals_110, (2048, 512), (512, 1))
    assert_size_stride(primals_111, (512, 2048), (2048, 1))
    assert_size_stride(primals_112, (512, 512), (512, 1))
    assert_size_stride(primals_113, (512, 512), (512, 1))
    assert_size_stride(primals_114, (512, 512), (512, 1))
    assert_size_stride(primals_115, (512, 512), (512, 1))
    assert_size_stride(primals_116, (512, 512), (512, 1))
    assert_size_stride(primals_117, (512, 512), (512, 1))
    assert_size_stride(primals_118, (512, 512), (512, 1))
    assert_size_stride(primals_119, (512, 512), (512, 1))
    assert_size_stride(primals_120, (2048, 512), (512, 1))
    assert_size_stride(primals_121, (512, 2048), (2048, 1))
    assert_size_stride(primals_122, (512, 512), (512, 1))
    assert_size_stride(primals_123, (512, 512), (512, 1))
    assert_size_stride(primals_124, (512, 512), (512, 1))
    assert_size_stride(primals_125, (512, 512), (512, 1))
    assert_size_stride(primals_126, (512, 512), (512, 1))
    assert_size_stride(primals_127, (512, 512), (512, 1))
    assert_size_stride(primals_128, (512, 512), (512, 1))
    assert_size_stride(primals_129, (512, 512), (512, 1))
    assert_size_stride(primals_130, (2048, 512), (512, 1))
    assert_size_stride(primals_131, (512, 2048), (2048, 1))
    assert_size_stride(primals_132, (32128, 512), (512, 1))
    assert_size_stride(primals_133, (1, 1024), (1024, 1))
    assert_size_stride(primals_134, (1, 1024), (1024, 1))
    assert_size_stride(primals_135, (1, 1024), (1024, 1))
    buf0 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_0(c_void_p(primals_133.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [hidden_states, inputs_embeds], Original ATen: [aten.embedding, aten.native_dropout]
    buf1 = aten.native_dropout(buf0, 0.1, True)
    buf2 = buf1[0]
    buf3 = buf1[1]
    del buf1
    buf4 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf4, (1, 1024, 1), (1024, 1, 1), 0); del buf4  # reuse
    buf6 = reinterpret_tensor(buf0, (1024, 512), (512, 1), 0); del buf0  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_1(c_void_p(buf5.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf6.data_ptr()))
    buf7 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_34, (512, 512), (1, 512), 0), out=buf7)
    buf8 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_35, (512, 512), (1, 512), 0), out=buf8)
    buf9 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_36, (512, 512), (1, 512), 0), out=buf9)
    buf10 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf8, (8, 64, 1024), (64, 1, 512), 0), out=buf10)
    buf11 = empty((1024, 1024), device='cpu', dtype=torch.int64)
    buf12 = empty_strided((1, 8, 1024, 1), (8192, 1024, 1, 8192), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf10, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf10  # reuse
    buf14 = empty_strided((1, 8, 1024, 1), (8192, 1024, 1, 8192), device='cpu', dtype=torch.float32)
    buf15 = buf13; del buf13  # reuse
    cpp_fused__softmax__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2(c_void_p(buf15.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    # Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.native_dropout]
    buf16 = aten.native_dropout(buf15, 0.1, True)
    buf17 = buf16[0]
    buf18 = buf16[1]
    del buf16
    buf19 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf17, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf9, (8, 1024, 64), (64, 512, 1), 0), out=buf19)
    buf20 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_3(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf19, (1024, 512), (512, 1), 0); del buf19  # reuse
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, reinterpret_tensor(primals_38, (512, 512), (1, 512), 0), out=buf21)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf22 = aten.native_dropout(reinterpret_tensor(buf21, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf25 = buf23; del buf23  # reuse
    buf26 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf26, (1, 1024, 1), (1024, 1, 1), 0); del buf26  # reuse
    buf28 = buf21; del buf21  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_4(c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((1024, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf28, reinterpret_tensor(primals_39, (512, 2048), (1, 512), 0), out=buf29)
    buf30 = reinterpret_tensor(buf29, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf29  # reuse
    buf581 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_5(c_void_p(buf30.data_ptr()), c_void_p(buf581.data_ptr()))
    # Source Nodes: [hidden_states_8, hidden_states_9], Original ATen: [aten.native_dropout, aten.relu]
    buf31 = aten.native_dropout(buf30, 0.1, True)
    buf32 = buf31[0]
    buf33 = buf31[1]
    del buf31
    buf34 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_40, (2048, 512), (1, 2048), 0), out=buf34)
    # Source Nodes: [l__mod___encoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf35 = aten.native_dropout(reinterpret_tensor(buf34, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = buf36; del buf36  # reuse
    buf39 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf40 = reinterpret_tensor(buf39, (1, 1024, 1), (1024, 1, 1), 0); del buf39  # reuse
    buf41 = buf34; del buf34  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_6(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf41, reinterpret_tensor(primals_41, (512, 512), (1, 512), 0), out=buf42)
    buf43 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf41, reinterpret_tensor(primals_42, (512, 512), (1, 512), 0), out=buf43)
    buf44 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf41, reinterpret_tensor(primals_43, (512, 512), (1, 512), 0), out=buf44)
    buf45 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf42, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf43, (8, 64, 1024), (64, 1, 512), 0), out=buf45)
    buf46 = buf14; del buf14  # reuse
    buf47 = reinterpret_tensor(buf45, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf45  # reuse
    buf48 = buf12; del buf12  # reuse
    buf49 = buf47; del buf47  # reuse
    cpp_fused__softmax_7(c_void_p(buf49.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()))
    # Source Nodes: [attn_weights_3, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
    buf50 = aten.native_dropout(buf49, 0.1, True)
    buf51 = buf50[0]
    buf52 = buf50[1]
    del buf50
    buf53 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf51, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf44, (8, 1024, 64), (64, 512, 1), 0), out=buf53)
    buf54 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    buf55 = reinterpret_tensor(buf53, (1024, 512), (512, 1), 0); del buf53  # reuse
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf54, reinterpret_tensor(primals_44, (512, 512), (1, 512), 0), out=buf55)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf56 = aten.native_dropout(reinterpret_tensor(buf55, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf57 = buf56[0]
    buf58 = buf56[1]
    del buf56
    buf59 = buf57; del buf57  # reuse
    buf60 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf61 = reinterpret_tensor(buf60, (1, 1024, 1), (1024, 1, 1), 0); del buf60  # reuse
    buf62 = buf55; del buf55  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_9(c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf30, (1024, 2048), (2048, 1), 0); del buf30  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf62, reinterpret_tensor(primals_45, (512, 2048), (1, 512), 0), out=buf63)
    buf64 = reinterpret_tensor(buf63, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf63  # reuse
    buf580 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_10(c_void_p(buf64.data_ptr()), c_void_p(buf580.data_ptr()))
    # Source Nodes: [hidden_states_21, hidden_states_22], Original ATen: [aten.native_dropout, aten.relu]
    buf65 = aten.native_dropout(buf64, 0.1, True)
    buf66 = buf65[0]
    buf67 = buf65[1]
    del buf65
    buf68 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_46, (2048, 512), (1, 2048), 0), out=buf68)
    # Source Nodes: [l__mod___encoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf69 = aten.native_dropout(reinterpret_tensor(buf68, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf70 = buf69[0]
    buf71 = buf69[1]
    del buf69
    buf72 = buf70; del buf70  # reuse
    buf73 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf73, (1, 1024, 1), (1024, 1, 1), 0); del buf73  # reuse
    buf75 = buf68; del buf68  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_11(c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, reinterpret_tensor(primals_47, (512, 512), (1, 512), 0), out=buf76)
    buf77 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, reinterpret_tensor(primals_48, (512, 512), (1, 512), 0), out=buf77)
    buf78 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), out=buf78)
    buf79 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf76, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf77, (8, 64, 1024), (64, 1, 512), 0), out=buf79)
    buf80 = buf48; del buf48  # reuse
    buf81 = reinterpret_tensor(buf79, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf79  # reuse
    buf82 = buf46; del buf46  # reuse
    buf83 = buf81; del buf81  # reuse
    cpp_fused__softmax_12(c_void_p(buf83.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    # Source Nodes: [attn_weights_5, softmax_2], Original ATen: [aten._softmax, aten.native_dropout]
    buf84 = aten.native_dropout(buf83, 0.1, True)
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf85, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf78, (8, 1024, 64), (64, 512, 1), 0), out=buf87)
    buf88 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_13(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf87, (1024, 512), (512, 1), 0); del buf87  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf88, reinterpret_tensor(primals_50, (512, 512), (1, 512), 0), out=buf89)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf90 = aten.native_dropout(reinterpret_tensor(buf89, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf91 = buf90[0]
    buf92 = buf90[1]
    del buf90
    buf93 = buf91; del buf91  # reuse
    buf94 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf94, (1, 1024, 1), (1024, 1, 1), 0); del buf94  # reuse
    buf96 = buf89; del buf89  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_14(c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = reinterpret_tensor(buf64, (1024, 2048), (2048, 1), 0); del buf64  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf96, reinterpret_tensor(primals_51, (512, 2048), (1, 512), 0), out=buf97)
    buf98 = reinterpret_tensor(buf97, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf97  # reuse
    buf579 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_15(c_void_p(buf98.data_ptr()), c_void_p(buf579.data_ptr()))
    # Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.native_dropout, aten.relu]
    buf99 = aten.native_dropout(buf98, 0.1, True)
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_52, (2048, 512), (1, 2048), 0), out=buf102)
    # Source Nodes: [l__mod___encoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf103 = aten.native_dropout(reinterpret_tensor(buf102, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf106 = buf104; del buf104  # reuse
    buf107 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf107, (1, 1024, 1), (1024, 1, 1), 0); del buf107  # reuse
    buf109 = buf102; del buf102  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_16(c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf109, reinterpret_tensor(primals_53, (512, 512), (1, 512), 0), out=buf110)
    buf111 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf109, reinterpret_tensor(primals_54, (512, 512), (1, 512), 0), out=buf111)
    buf112 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf109, reinterpret_tensor(primals_55, (512, 512), (1, 512), 0), out=buf112)
    buf113 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf111, (8, 64, 1024), (64, 1, 512), 0), out=buf113)
    buf114 = buf82; del buf82  # reuse
    buf115 = reinterpret_tensor(buf113, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf113  # reuse
    buf116 = buf80; del buf80  # reuse
    buf117 = buf115; del buf115  # reuse
    cpp_fused__softmax_17(c_void_p(buf117.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    # Source Nodes: [attn_weights_7, softmax_3], Original ATen: [aten._softmax, aten.native_dropout]
    buf118 = aten.native_dropout(buf117, 0.1, True)
    buf119 = buf118[0]
    buf120 = buf118[1]
    del buf118
    buf121 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf112, (8, 1024, 64), (64, 512, 1), 0), out=buf121)
    buf122 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_18(c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf121, (1024, 512), (512, 1), 0); del buf121  # reuse
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, reinterpret_tensor(primals_56, (512, 512), (1, 512), 0), out=buf123)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf124 = aten.native_dropout(reinterpret_tensor(buf123, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf125 = buf124[0]
    buf126 = buf124[1]
    del buf124
    buf127 = buf125; del buf125  # reuse
    buf128 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf129 = reinterpret_tensor(buf128, (1, 1024, 1), (1024, 1, 1), 0); del buf128  # reuse
    buf130 = buf123; del buf123  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_19(c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = reinterpret_tensor(buf98, (1024, 2048), (2048, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf130, reinterpret_tensor(primals_57, (512, 2048), (1, 512), 0), out=buf131)
    buf132 = reinterpret_tensor(buf131, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf131  # reuse
    buf578 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_20(c_void_p(buf132.data_ptr()), c_void_p(buf578.data_ptr()))
    # Source Nodes: [hidden_states_47, hidden_states_48], Original ATen: [aten.native_dropout, aten.relu]
    buf133 = aten.native_dropout(buf132, 0.1, True)
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf136 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_58, (2048, 512), (1, 2048), 0), out=buf136)
    # Source Nodes: [l__mod___encoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf137 = aten.native_dropout(reinterpret_tensor(buf136, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf138 = buf137[0]
    buf139 = buf137[1]
    del buf137
    buf140 = buf138; del buf138  # reuse
    buf141 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf142 = reinterpret_tensor(buf141, (1, 1024, 1), (1024, 1, 1), 0); del buf141  # reuse
    buf143 = buf136; del buf136  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_21(c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf143, reinterpret_tensor(primals_59, (512, 512), (1, 512), 0), out=buf144)
    buf145 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf143, reinterpret_tensor(primals_60, (512, 512), (1, 512), 0), out=buf145)
    buf146 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf143, reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), out=buf146)
    buf147 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf145, (8, 64, 1024), (64, 1, 512), 0), out=buf147)
    buf148 = buf116; del buf116  # reuse
    buf149 = reinterpret_tensor(buf147, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf147  # reuse
    buf150 = buf114; del buf114  # reuse
    buf151 = buf149; del buf149  # reuse
    cpp_fused__softmax_22(c_void_p(buf151.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    # Source Nodes: [attn_weights_9, softmax_4], Original ATen: [aten._softmax, aten.native_dropout]
    buf152 = aten.native_dropout(buf151, 0.1, True)
    buf153 = buf152[0]
    buf154 = buf152[1]
    del buf152
    buf155 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf153, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf146, (8, 1024, 64), (64, 512, 1), 0), out=buf155)
    buf156 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_23(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf155, (1024, 512), (512, 1), 0); del buf155  # reuse
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf156, reinterpret_tensor(primals_62, (512, 512), (1, 512), 0), out=buf157)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf158 = aten.native_dropout(reinterpret_tensor(buf157, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = buf159; del buf159  # reuse
    buf162 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf162, (1, 1024, 1), (1024, 1, 1), 0); del buf162  # reuse
    buf164 = buf157; del buf157  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_24(c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf132, (1024, 2048), (2048, 1), 0); del buf132  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
    extern_kernels.mm(buf164, reinterpret_tensor(primals_63, (512, 2048), (1, 512), 0), out=buf165)
    buf166 = reinterpret_tensor(buf165, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf165  # reuse
    buf577 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_25(c_void_p(buf166.data_ptr()), c_void_p(buf577.data_ptr()))
    # Source Nodes: [hidden_states_60, hidden_states_61], Original ATen: [aten.native_dropout, aten.relu]
    buf167 = aten.native_dropout(buf166, 0.1, True)
    buf168 = buf167[0]
    buf169 = buf167[1]
    del buf167
    buf170 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_64, (2048, 512), (1, 2048), 0), out=buf170)
    # Source Nodes: [l__mod___encoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf171 = aten.native_dropout(reinterpret_tensor(buf170, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = buf172; del buf172  # reuse
    buf175 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf176 = reinterpret_tensor(buf175, (1, 1024, 1), (1024, 1, 1), 0); del buf175  # reuse
    buf177 = buf170; del buf170  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_26(c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf177, reinterpret_tensor(primals_65, (512, 512), (1, 512), 0), out=buf178)
    buf179 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf177, reinterpret_tensor(primals_66, (512, 512), (1, 512), 0), out=buf179)
    buf180 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf177, reinterpret_tensor(primals_67, (512, 512), (1, 512), 0), out=buf180)
    buf181 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf178, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf179, (8, 64, 1024), (64, 1, 512), 0), out=buf181)
    buf182 = buf150; del buf150  # reuse
    buf183 = reinterpret_tensor(buf181, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf181  # reuse
    buf184 = buf148; del buf148  # reuse
    buf185 = buf183; del buf183  # reuse
    cpp_fused__softmax_27(c_void_p(buf185.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()))
    del primals_37
    # Source Nodes: [attn_weights_11, softmax_5], Original ATen: [aten._softmax, aten.native_dropout]
    buf186 = aten.native_dropout(buf185, 0.1, True)
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf189 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf187, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf180, (8, 1024, 64), (64, 512, 1), 0), out=buf189)
    buf190 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_28(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf189, (1024, 512), (512, 1), 0); del buf189  # reuse
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf190, reinterpret_tensor(primals_68, (512, 512), (1, 512), 0), out=buf191)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf192 = aten.native_dropout(reinterpret_tensor(buf191, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf193 = buf192[0]
    buf194 = buf192[1]
    del buf192
    buf195 = buf193; del buf193  # reuse
    buf196 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf197 = reinterpret_tensor(buf196, (1, 1024, 1), (1024, 1, 1), 0); del buf196  # reuse
    buf198 = buf191; del buf191  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_29(c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = reinterpret_tensor(buf166, (1024, 2048), (2048, 1), 0); del buf166  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
    extern_kernels.mm(buf198, reinterpret_tensor(primals_69, (512, 2048), (1, 512), 0), out=buf199)
    buf200 = reinterpret_tensor(buf199, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf199  # reuse
    buf576 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_30(c_void_p(buf200.data_ptr()), c_void_p(buf576.data_ptr()))
    # Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.native_dropout, aten.relu]
    buf201 = aten.native_dropout(buf200, 0.1, True)
    buf202 = buf201[0]
    buf203 = buf201[1]
    del buf201
    buf204 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_70, (2048, 512), (1, 2048), 0), out=buf204)
    # Source Nodes: [l__mod___encoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf205 = aten.native_dropout(reinterpret_tensor(buf204, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf206 = buf205[0]
    buf207 = buf205[1]
    del buf205
    buf208 = buf206; del buf206  # reuse
    buf209 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf210 = reinterpret_tensor(buf209, (1, 1024, 1), (1024, 1, 1), 0); del buf209  # reuse
    buf211 = reinterpret_tensor(buf204, (1, 1024, 512), (524288, 512, 1), 0); del buf204  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_31(c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf211.data_ptr()))
    # Source Nodes: [hidden_states_79, hidden_states_80, hidden_states_82], Original ATen: [aten.mul, aten.native_dropout]
    buf212 = aten.native_dropout(buf211, 0.1, True)
    buf213 = buf212[0]
    buf214 = buf212[1]
    del buf212
    buf215 = buf211; del buf211  # reuse
    cpp_fused_embedding_32(c_void_p(primals_135.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf215.data_ptr()))
    del primals_33
    # Source Nodes: [hidden_states_83, inputs_embeds_1], Original ATen: [aten.embedding, aten.native_dropout]
    buf216 = aten.native_dropout(buf215, 0.1, True)
    buf217 = buf216[0]
    buf218 = buf216[1]
    del buf216
    buf219 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf219, (1, 1024, 1), (1024, 1, 1), 0); del buf219  # reuse
    buf221 = reinterpret_tensor(buf215, (1024, 512), (512, 1), 0); del buf215  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_33(c_void_p(buf220.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf221, reinterpret_tensor(primals_71, (512, 512), (1, 512), 0), out=buf222)
    buf223 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf221, reinterpret_tensor(primals_72, (512, 512), (1, 512), 0), out=buf223)
    buf224 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf221, reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), out=buf224)
    buf225 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf222, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf223, (8, 64, 1024), (64, 1, 512), 0), out=buf225)
    buf226 = empty((1024, 1024), device='cpu', dtype=torch.int64)
    buf227 = buf184; del buf184  # reuse
    buf228 = reinterpret_tensor(buf225, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf225  # reuse
    buf229 = buf228; del buf228  # reuse
    buf230 = buf182; del buf182  # reuse
    buf231 = buf229; del buf229  # reuse
    cpp_fused__softmax__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_34(c_void_p(buf231.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf230.data_ptr()))
    # Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.native_dropout]
    buf232 = aten.native_dropout(buf231, 0.1, True)
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf224, (8, 1024, 64), (64, 512, 1), 0), out=buf235)
    buf236 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_35(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf235, (1024, 512), (512, 1), 0); del buf235  # reuse
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf236, reinterpret_tensor(primals_75, (512, 512), (1, 512), 0), out=buf237)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf238 = aten.native_dropout(reinterpret_tensor(buf237, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    buf241 = buf239; del buf239  # reuse
    buf242 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf242, (1, 1024, 1), (1024, 1, 1), 0); del buf242  # reuse
    buf244 = buf237; del buf237  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_36(c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf244.data_ptr()))
    buf245 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf244, reinterpret_tensor(primals_76, (512, 512), (1, 512), 0), out=buf245)
    buf246 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (1, 512), 0), out=buf246)
    buf247 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (1, 512), 0), out=buf247)
    buf248 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf245, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf246, (8, 64, 1024), (64, 1, 512), 0), out=buf248)
    buf249 = buf230; del buf230  # reuse
    buf250 = reinterpret_tensor(buf248, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf248  # reuse
    buf251 = buf227; del buf227  # reuse
    buf252 = buf250; del buf250  # reuse
    cpp_fused__softmax_37(c_void_p(buf252.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    # Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.native_dropout]
    buf253 = aten.native_dropout(buf252, 0.1, True)
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf247, (8, 1024, 64), (64, 512, 1), 0), out=buf256)
    buf257 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_38(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    buf258 = reinterpret_tensor(buf256, (1024, 512), (512, 1), 0); del buf256  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf257, reinterpret_tensor(primals_79, (512, 512), (1, 512), 0), out=buf258)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf259 = aten.native_dropout(reinterpret_tensor(buf258, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf260 = buf259[0]
    buf261 = buf259[1]
    del buf259
    buf262 = buf260; del buf260  # reuse
    buf263 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf264 = reinterpret_tensor(buf263, (1, 1024, 1), (1024, 1, 1), 0); del buf263  # reuse
    buf265 = buf258; del buf258  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_39(c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = reinterpret_tensor(buf200, (1024, 2048), (2048, 1), 0); del buf200  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
    extern_kernels.mm(buf265, reinterpret_tensor(primals_80, (512, 2048), (1, 512), 0), out=buf266)
    buf267 = reinterpret_tensor(buf266, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf266  # reuse
    buf575 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_40(c_void_p(buf267.data_ptr()), c_void_p(buf575.data_ptr()))
    # Source Nodes: [hidden_states_95, hidden_states_96], Original ATen: [aten.native_dropout, aten.relu]
    buf268 = aten.native_dropout(buf267, 0.1, True)
    buf269 = buf268[0]
    buf270 = buf268[1]
    del buf268
    buf271 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_81, (2048, 512), (1, 2048), 0), out=buf271)
    # Source Nodes: [l__mod___decoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf272 = aten.native_dropout(reinterpret_tensor(buf271, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf273 = buf272[0]
    buf274 = buf272[1]
    del buf272
    buf275 = buf273; del buf273  # reuse
    buf276 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf277 = reinterpret_tensor(buf276, (1, 1024, 1), (1024, 1, 1), 0); del buf276  # reuse
    buf278 = buf271; del buf271  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_41(c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf278.data_ptr()))
    buf279 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf278, reinterpret_tensor(primals_82, (512, 512), (1, 512), 0), out=buf279)
    buf280 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf278, reinterpret_tensor(primals_83, (512, 512), (1, 512), 0), out=buf280)
    buf281 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf278, reinterpret_tensor(primals_84, (512, 512), (1, 512), 0), out=buf281)
    buf282 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf280, (8, 64, 1024), (64, 1, 512), 0), out=buf282)
    buf283 = buf251; del buf251  # reuse
    buf284 = reinterpret_tensor(buf282, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf282  # reuse
    buf285 = buf284; del buf284  # reuse
    buf286 = buf249; del buf249  # reuse
    buf287 = buf285; del buf285  # reuse
    cpp_fused__softmax_42(c_void_p(buf287.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf286.data_ptr()))
    # Source Nodes: [attn_weights_17, softmax_8], Original ATen: [aten._softmax, aten.native_dropout]
    buf288 = aten.native_dropout(buf287, 0.1, True)
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf289, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf281, (8, 1024, 64), (64, 512, 1), 0), out=buf291)
    buf292 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_43(c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    buf293 = reinterpret_tensor(buf291, (1024, 512), (512, 1), 0); del buf291  # reuse
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf292, reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), out=buf293)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf294 = aten.native_dropout(reinterpret_tensor(buf293, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf295 = buf294[0]
    buf296 = buf294[1]
    del buf294
    buf297 = buf295; del buf295  # reuse
    buf298 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf298, (1, 1024, 1), (1024, 1, 1), 0); del buf298  # reuse
    buf300 = buf293; del buf293  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_44(c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf300.data_ptr()))
    buf301 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf300, reinterpret_tensor(primals_86, (512, 512), (1, 512), 0), out=buf301)
    buf302 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (1, 512), 0), out=buf302)
    buf303 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (1, 512), 0), out=buf303)
    buf304 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf302, (8, 64, 1024), (64, 1, 512), 0), out=buf304)
    buf305 = buf286; del buf286  # reuse
    buf306 = reinterpret_tensor(buf304, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf304  # reuse
    buf307 = buf283; del buf283  # reuse
    buf308 = buf306; del buf306  # reuse
    cpp_fused__softmax_45(c_void_p(buf308.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()))
    # Source Nodes: [attn_weights_19, softmax_9], Original ATen: [aten._softmax, aten.native_dropout]
    buf309 = aten.native_dropout(buf308, 0.1, True)
    buf310 = buf309[0]
    buf311 = buf309[1]
    del buf309
    buf312 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf303, (8, 1024, 64), (64, 512, 1), 0), out=buf312)
    buf313 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_46(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf312, (1024, 512), (512, 1), 0); del buf312  # reuse
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(buf313, reinterpret_tensor(primals_89, (512, 512), (1, 512), 0), out=buf314)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf315 = aten.native_dropout(reinterpret_tensor(buf314, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf316 = buf315[0]
    buf317 = buf315[1]
    del buf315
    buf318 = buf316; del buf316  # reuse
    buf319 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf320 = reinterpret_tensor(buf319, (1, 1024, 1), (1024, 1, 1), 0); del buf319  # reuse
    buf321 = buf314; del buf314  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_47(c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = reinterpret_tensor(buf267, (1024, 2048), (2048, 1), 0); del buf267  # reuse
    # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
    extern_kernels.mm(buf321, reinterpret_tensor(primals_90, (512, 2048), (1, 512), 0), out=buf322)
    buf323 = reinterpret_tensor(buf322, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf322  # reuse
    buf574 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_48(c_void_p(buf323.data_ptr()), c_void_p(buf574.data_ptr()))
    # Source Nodes: [hidden_states_112, hidden_states_113], Original ATen: [aten.native_dropout, aten.relu]
    buf324 = aten.native_dropout(buf323, 0.1, True)
    buf325 = buf324[0]
    buf326 = buf324[1]
    del buf324
    buf327 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), out=buf327)
    # Source Nodes: [l__mod___decoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf328 = aten.native_dropout(reinterpret_tensor(buf327, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = buf329; del buf329  # reuse
    buf332 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf333 = reinterpret_tensor(buf332, (1, 1024, 1), (1024, 1, 1), 0); del buf332  # reuse
    buf334 = buf327; del buf327  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_49(c_void_p(buf331.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf334.data_ptr()))
    buf335 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf334, reinterpret_tensor(primals_92, (512, 512), (1, 512), 0), out=buf335)
    buf336 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf334, reinterpret_tensor(primals_93, (512, 512), (1, 512), 0), out=buf336)
    buf337 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf334, reinterpret_tensor(primals_94, (512, 512), (1, 512), 0), out=buf337)
    buf338 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf336, (8, 64, 1024), (64, 1, 512), 0), out=buf338)
    buf339 = buf307; del buf307  # reuse
    buf340 = reinterpret_tensor(buf338, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf338  # reuse
    buf341 = buf340; del buf340  # reuse
    buf342 = buf305; del buf305  # reuse
    buf343 = buf341; del buf341  # reuse
    cpp_fused__softmax_50(c_void_p(buf343.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf342.data_ptr()))
    # Source Nodes: [attn_weights_21, softmax_10], Original ATen: [aten._softmax, aten.native_dropout]
    buf344 = aten.native_dropout(buf343, 0.1, True)
    buf345 = buf344[0]
    buf346 = buf344[1]
    del buf344
    buf347 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf345, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf337, (8, 1024, 64), (64, 512, 1), 0), out=buf347)
    buf348 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_51(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = reinterpret_tensor(buf347, (1024, 512), (512, 1), 0); del buf347  # reuse
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf348, reinterpret_tensor(primals_95, (512, 512), (1, 512), 0), out=buf349)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf350 = aten.native_dropout(reinterpret_tensor(buf349, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf351 = buf350[0]
    buf352 = buf350[1]
    del buf350
    buf353 = buf351; del buf351  # reuse
    buf354 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf355 = reinterpret_tensor(buf354, (1, 1024, 1), (1024, 1, 1), 0); del buf354  # reuse
    buf356 = buf349; del buf349  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_52(c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf356, reinterpret_tensor(primals_96, (512, 512), (1, 512), 0), out=buf357)
    buf358 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), out=buf358)
    buf359 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (1, 512), 0), out=buf359)
    buf360 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf357, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf358, (8, 64, 1024), (64, 1, 512), 0), out=buf360)
    buf361 = buf342; del buf342  # reuse
    buf362 = reinterpret_tensor(buf360, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf360  # reuse
    buf363 = buf339; del buf339  # reuse
    buf364 = buf362; del buf362  # reuse
    cpp_fused__softmax_53(c_void_p(buf364.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()))
    # Source Nodes: [attn_weights_23, softmax_11], Original ATen: [aten._softmax, aten.native_dropout]
    buf365 = aten.native_dropout(buf364, 0.1, True)
    buf366 = buf365[0]
    buf367 = buf365[1]
    del buf365
    buf368 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf366, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf359, (8, 1024, 64), (64, 512, 1), 0), out=buf368)
    buf369 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_54(c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()))
    buf370 = reinterpret_tensor(buf368, (1024, 512), (512, 1), 0); del buf368  # reuse
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(buf369, reinterpret_tensor(primals_99, (512, 512), (1, 512), 0), out=buf370)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf371 = aten.native_dropout(reinterpret_tensor(buf370, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf372 = buf371[0]
    buf373 = buf371[1]
    del buf371
    buf374 = buf372; del buf372  # reuse
    buf375 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf376 = reinterpret_tensor(buf375, (1, 1024, 1), (1024, 1, 1), 0); del buf375  # reuse
    buf377 = buf370; del buf370  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_55(c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = reinterpret_tensor(buf323, (1024, 2048), (2048, 1), 0); del buf323  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
    extern_kernels.mm(buf377, reinterpret_tensor(primals_100, (512, 2048), (1, 512), 0), out=buf378)
    buf379 = reinterpret_tensor(buf378, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf378  # reuse
    buf573 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_56(c_void_p(buf379.data_ptr()), c_void_p(buf573.data_ptr()))
    # Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.native_dropout, aten.relu]
    buf380 = aten.native_dropout(buf379, 0.1, True)
    buf381 = buf380[0]
    buf382 = buf380[1]
    del buf380
    buf383 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_101, (2048, 512), (1, 2048), 0), out=buf383)
    # Source Nodes: [l__mod___decoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf384 = aten.native_dropout(reinterpret_tensor(buf383, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf385 = buf384[0]
    buf386 = buf384[1]
    del buf384
    buf387 = buf385; del buf385  # reuse
    buf388 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf389 = reinterpret_tensor(buf388, (1, 1024, 1), (1024, 1, 1), 0); del buf388  # reuse
    buf390 = buf383; del buf383  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_57(c_void_p(buf387.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, reinterpret_tensor(primals_102, (512, 512), (1, 512), 0), out=buf391)
    buf392 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, reinterpret_tensor(primals_103, (512, 512), (1, 512), 0), out=buf392)
    buf393 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, reinterpret_tensor(primals_104, (512, 512), (1, 512), 0), out=buf393)
    buf394 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf391, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf392, (8, 64, 1024), (64, 1, 512), 0), out=buf394)
    buf395 = buf363; del buf363  # reuse
    buf396 = reinterpret_tensor(buf394, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf394  # reuse
    buf397 = buf396; del buf396  # reuse
    buf398 = buf361; del buf361  # reuse
    buf399 = buf397; del buf397  # reuse
    cpp_fused__softmax_58(c_void_p(buf399.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf398.data_ptr()))
    # Source Nodes: [attn_weights_25, softmax_12], Original ATen: [aten._softmax, aten.native_dropout]
    buf400 = aten.native_dropout(buf399, 0.1, True)
    buf401 = buf400[0]
    buf402 = buf400[1]
    del buf400
    buf403 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf401, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf393, (8, 1024, 64), (64, 512, 1), 0), out=buf403)
    buf404 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_59(c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    buf405 = reinterpret_tensor(buf403, (1024, 512), (512, 1), 0); del buf403  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(buf404, reinterpret_tensor(primals_105, (512, 512), (1, 512), 0), out=buf405)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf406 = aten.native_dropout(reinterpret_tensor(buf405, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf407 = buf406[0]
    buf408 = buf406[1]
    del buf406
    buf409 = buf407; del buf407  # reuse
    buf410 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf411 = reinterpret_tensor(buf410, (1, 1024, 1), (1024, 1, 1), 0); del buf410  # reuse
    buf412 = buf405; del buf405  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_60(c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf412.data_ptr()))
    buf413 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf412, reinterpret_tensor(primals_106, (512, 512), (1, 512), 0), out=buf413)
    buf414 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (1, 512), 0), out=buf414)
    buf415 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (1, 512), 0), out=buf415)
    buf416 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf413, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf414, (8, 64, 1024), (64, 1, 512), 0), out=buf416)
    buf417 = buf398; del buf398  # reuse
    buf418 = reinterpret_tensor(buf416, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf416  # reuse
    buf419 = buf395; del buf395  # reuse
    buf420 = buf418; del buf418  # reuse
    cpp_fused__softmax_61(c_void_p(buf420.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()))
    # Source Nodes: [attn_weights_27, softmax_13], Original ATen: [aten._softmax, aten.native_dropout]
    buf421 = aten.native_dropout(buf420, 0.1, True)
    buf422 = buf421[0]
    buf423 = buf421[1]
    del buf421
    buf424 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf422, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf415, (8, 1024, 64), (64, 512, 1), 0), out=buf424)
    buf425 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_62(c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf424, (1024, 512), (512, 1), 0); del buf424  # reuse
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf425, reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), out=buf426)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf427 = aten.native_dropout(reinterpret_tensor(buf426, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf428 = buf427[0]
    buf429 = buf427[1]
    del buf427
    buf430 = buf428; del buf428  # reuse
    buf431 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf432 = reinterpret_tensor(buf431, (1, 1024, 1), (1024, 1, 1), 0); del buf431  # reuse
    buf433 = buf426; del buf426  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_63(c_void_p(buf430.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = reinterpret_tensor(buf379, (1024, 2048), (2048, 1), 0); del buf379  # reuse
    # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
    extern_kernels.mm(buf433, reinterpret_tensor(primals_110, (512, 2048), (1, 512), 0), out=buf434)
    buf435 = reinterpret_tensor(buf434, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf434  # reuse
    buf572 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_64(c_void_p(buf435.data_ptr()), c_void_p(buf572.data_ptr()))
    # Source Nodes: [hidden_states_146, hidden_states_147], Original ATen: [aten.native_dropout, aten.relu]
    buf436 = aten.native_dropout(buf435, 0.1, True)
    buf437 = buf436[0]
    buf438 = buf436[1]
    del buf436
    buf439 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf437, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_111, (2048, 512), (1, 2048), 0), out=buf439)
    # Source Nodes: [l__mod___decoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf440 = aten.native_dropout(reinterpret_tensor(buf439, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf441 = buf440[0]
    buf442 = buf440[1]
    del buf440
    buf443 = buf441; del buf441  # reuse
    buf444 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf445 = reinterpret_tensor(buf444, (1, 1024, 1), (1024, 1, 1), 0); del buf444  # reuse
    buf446 = buf439; del buf439  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_65(c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf446.data_ptr()))
    buf447 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf446, reinterpret_tensor(primals_112, (512, 512), (1, 512), 0), out=buf447)
    buf448 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf446, reinterpret_tensor(primals_113, (512, 512), (1, 512), 0), out=buf448)
    buf449 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf446, reinterpret_tensor(primals_114, (512, 512), (1, 512), 0), out=buf449)
    buf450 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf447, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf448, (8, 64, 1024), (64, 1, 512), 0), out=buf450)
    buf451 = buf419; del buf419  # reuse
    buf452 = reinterpret_tensor(buf450, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf450  # reuse
    buf453 = buf452; del buf452  # reuse
    buf454 = buf417; del buf417  # reuse
    buf455 = buf453; del buf453  # reuse
    cpp_fused__softmax_66(c_void_p(buf455.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf454.data_ptr()))
    # Source Nodes: [attn_weights_29, softmax_14], Original ATen: [aten._softmax, aten.native_dropout]
    buf456 = aten.native_dropout(buf455, 0.1, True)
    buf457 = buf456[0]
    buf458 = buf456[1]
    del buf456
    buf459 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf457, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf449, (8, 1024, 64), (64, 512, 1), 0), out=buf459)
    buf460 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_67(c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    buf461 = reinterpret_tensor(buf459, (1024, 512), (512, 1), 0); del buf459  # reuse
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf460, reinterpret_tensor(primals_115, (512, 512), (1, 512), 0), out=buf461)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf462 = aten.native_dropout(reinterpret_tensor(buf461, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf463 = buf462[0]
    buf464 = buf462[1]
    del buf462
    buf465 = buf463; del buf463  # reuse
    buf466 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf467 = reinterpret_tensor(buf466, (1, 1024, 1), (1024, 1, 1), 0); del buf466  # reuse
    buf468 = buf461; del buf461  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_68(c_void_p(buf465.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf468.data_ptr()))
    buf469 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf468, reinterpret_tensor(primals_116, (512, 512), (1, 512), 0), out=buf469)
    buf470 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (1, 512), 0), out=buf470)
    buf471 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (1, 512), 0), out=buf471)
    buf472 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf469, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf470, (8, 64, 1024), (64, 1, 512), 0), out=buf472)
    buf473 = buf454; del buf454  # reuse
    buf474 = reinterpret_tensor(buf472, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf472  # reuse
    buf475 = buf451; del buf451  # reuse
    buf476 = buf474; del buf474  # reuse
    cpp_fused__softmax_69(c_void_p(buf476.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()))
    # Source Nodes: [attn_weights_31, softmax_15], Original ATen: [aten._softmax, aten.native_dropout]
    buf477 = aten.native_dropout(buf476, 0.1, True)
    buf478 = buf477[0]
    buf479 = buf477[1]
    del buf477
    buf480 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf478, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf471, (8, 1024, 64), (64, 512, 1), 0), out=buf480)
    buf481 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_70(c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    buf482 = reinterpret_tensor(buf480, (1024, 512), (512, 1), 0); del buf480  # reuse
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(buf481, reinterpret_tensor(primals_119, (512, 512), (1, 512), 0), out=buf482)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf483 = aten.native_dropout(reinterpret_tensor(buf482, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf484 = buf483[0]
    buf485 = buf483[1]
    del buf483
    buf486 = buf484; del buf484  # reuse
    buf487 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf488 = reinterpret_tensor(buf487, (1, 1024, 1), (1024, 1, 1), 0); del buf487  # reuse
    buf489 = buf482; del buf482  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_71(c_void_p(buf486.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = reinterpret_tensor(buf435, (1024, 2048), (2048, 1), 0); del buf435  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
    extern_kernels.mm(buf489, reinterpret_tensor(primals_120, (512, 2048), (1, 512), 0), out=buf490)
    buf491 = reinterpret_tensor(buf490, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf490  # reuse
    buf571 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_72(c_void_p(buf491.data_ptr()), c_void_p(buf571.data_ptr()))
    # Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.native_dropout, aten.relu]
    buf492 = aten.native_dropout(buf491, 0.1, True)
    buf493 = buf492[0]
    buf494 = buf492[1]
    del buf492
    buf495 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_121, (2048, 512), (1, 2048), 0), out=buf495)
    # Source Nodes: [l__mod___decoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf496 = aten.native_dropout(reinterpret_tensor(buf495, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf497 = buf496[0]
    buf498 = buf496[1]
    del buf496
    buf499 = buf497; del buf497  # reuse
    buf500 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf501 = reinterpret_tensor(buf500, (1, 1024, 1), (1024, 1, 1), 0); del buf500  # reuse
    buf502 = buf495; del buf495  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_73(c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf502.data_ptr()))
    buf503 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf502, reinterpret_tensor(primals_122, (512, 512), (1, 512), 0), out=buf503)
    buf504 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf502, reinterpret_tensor(primals_123, (512, 512), (1, 512), 0), out=buf504)
    buf505 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf502, reinterpret_tensor(primals_124, (512, 512), (1, 512), 0), out=buf505)
    buf506 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf503, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf504, (8, 64, 1024), (64, 1, 512), 0), out=buf506)
    buf507 = buf475; del buf475  # reuse
    buf508 = reinterpret_tensor(buf506, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf506  # reuse
    buf509 = buf508; del buf508  # reuse
    buf510 = buf473; del buf473  # reuse
    buf511 = buf509; del buf509  # reuse
    cpp_fused__softmax_74(c_void_p(buf511.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf510.data_ptr()))
    del primals_74
    # Source Nodes: [attn_weights_33, softmax_16], Original ATen: [aten._softmax, aten.native_dropout]
    buf512 = aten.native_dropout(buf511, 0.1, True)
    buf513 = buf512[0]
    buf514 = buf512[1]
    del buf512
    buf515 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf513, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf505, (8, 1024, 64), (64, 512, 1), 0), out=buf515)
    buf516 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_75(c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    buf517 = reinterpret_tensor(buf515, (1024, 512), (512, 1), 0); del buf515  # reuse
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf516, reinterpret_tensor(primals_125, (512, 512), (1, 512), 0), out=buf517)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf518 = aten.native_dropout(reinterpret_tensor(buf517, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf519 = buf518[0]
    buf520 = buf518[1]
    del buf518
    buf521 = buf519; del buf519  # reuse
    buf522 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf523 = reinterpret_tensor(buf522, (1, 1024, 1), (1024, 1, 1), 0); del buf522  # reuse
    buf524 = buf517; del buf517  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_76(c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf524, reinterpret_tensor(primals_126, (512, 512), (1, 512), 0), out=buf525)
    buf526 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (1, 512), 0), out=buf526)
    buf527 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (1, 512), 0), out=buf527)
    buf528 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf525, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf526, (8, 64, 1024), (64, 1, 512), 0), out=buf528)
    buf529 = buf510; del buf510  # reuse
    buf530 = reinterpret_tensor(buf528, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf528  # reuse
    buf531 = buf507; del buf507  # reuse
    buf532 = buf530; del buf530  # reuse
    cpp_fused__softmax_77(c_void_p(buf532.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()))
    del buf529
    del buf531
    # Source Nodes: [attn_weights_35, softmax_17], Original ATen: [aten._softmax, aten.native_dropout]
    buf533 = aten.native_dropout(buf532, 0.1, True)
    buf534 = buf533[0]
    buf535 = buf533[1]
    del buf533
    buf536 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf534, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf527, (8, 1024, 64), (64, 512, 1), 0), out=buf536)
    buf537 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_78(c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()))
    buf538 = reinterpret_tensor(buf536, (1024, 512), (512, 1), 0); del buf536  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(buf537, reinterpret_tensor(primals_129, (512, 512), (1, 512), 0), out=buf538)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf539 = aten.native_dropout(reinterpret_tensor(buf538, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf540 = buf539[0]
    buf541 = buf539[1]
    del buf539
    buf542 = buf540; del buf540  # reuse
    buf543 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf544 = reinterpret_tensor(buf543, (1, 1024, 1), (1024, 1, 1), 0); del buf543  # reuse
    buf545 = buf538; del buf538  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_79(c_void_p(buf542.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf545.data_ptr()))
    buf546 = reinterpret_tensor(buf491, (1024, 2048), (2048, 1), 0); del buf491  # reuse
    # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
    extern_kernels.mm(buf545, reinterpret_tensor(primals_130, (512, 2048), (1, 512), 0), out=buf546)
    buf547 = reinterpret_tensor(buf546, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf546  # reuse
    buf570 = empty((1, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_80(c_void_p(buf547.data_ptr()), c_void_p(buf570.data_ptr()))
    # Source Nodes: [hidden_states_180, hidden_states_181], Original ATen: [aten.native_dropout, aten.relu]
    buf548 = aten.native_dropout(buf547, 0.1, True)
    del buf547
    buf549 = buf548[0]
    buf550 = buf548[1]
    del buf548
    buf551 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf549, (1024, 2048), (2048, 1), 0), reinterpret_tensor(primals_131, (2048, 512), (1, 2048), 0), out=buf551)
    # Source Nodes: [l__mod___decoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf552 = aten.native_dropout(reinterpret_tensor(buf551, (1, 1024, 512), (524288, 512, 1), 0), 0.1, True)
    buf553 = buf552[0]
    buf554 = buf552[1]
    del buf552
    buf555 = buf553; del buf553  # reuse
    buf556 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf557 = reinterpret_tensor(buf556, (1, 1024, 1), (1024, 1, 1), 0); del buf556  # reuse
    buf558 = reinterpret_tensor(buf551, (1, 1024, 512), (524288, 512, 1), 0); del buf551  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_81(c_void_p(buf555.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf558.data_ptr()))
    # Source Nodes: [hidden_states_186, hidden_states_187, sequence_output], Original ATen: [aten.mul, aten.native_dropout]
    buf559 = aten.native_dropout(buf558, 0.1, True)
    del buf558
    buf560 = buf559[0]
    buf561 = buf559[1]
    del buf559
    buf562 = reinterpret_tensor(buf560, (1024, 512), (512, 1), 0); del buf560  # reuse
    cpp_fused_mul_view_82(c_void_p(buf562.data_ptr()))
    buf563 = empty((1024, 32128), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(buf562, reinterpret_tensor(primals_132, (512, 32128), (1, 512), 0), out=buf563)
    buf564 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.float32)
    buf565 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.float32)
    buf566 = empty((1024, 32128), device='cpu', dtype=torch.float32)
    buf567 = empty((), device='cpu', dtype=torch.int64)
    buf569 = empty((), device='cpu', dtype=torch.float32)
    buf568 = empty((), device='cpu', dtype=torch.float32)
    buf582 = buf569; del buf569  # reuse
    cpp_fused__log_softmax_nll_loss_forward_83(c_void_p(buf582.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    return (buf582, reinterpret_tensor(buf563, (1, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf223, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf224, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf246, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf247, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf280, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf281, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf302, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf303, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf336, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf337, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf358, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf359, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf392, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf393, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf414, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf415, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf448, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf449, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf470, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf471, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf504, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf505, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf526, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf527, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), buf213, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, primals_133, buf2, buf3, buf5, buf6, buf11, buf18, buf20, buf24, buf25, buf27, buf28, buf33, reinterpret_tensor(buf32, (1024, 2048), (2048, 1), 0), buf37, buf38, buf40, buf41, buf52, buf54, buf58, buf59, buf61, buf62, buf67, reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0), buf71, buf72, buf74, buf75, buf86, buf88, buf92, buf93, buf95, buf96, buf101, reinterpret_tensor(buf100, (1024, 2048), (2048, 1), 0), buf105, buf106, buf108, buf109, buf120, buf122, buf126, buf127, buf129, buf130, buf135, reinterpret_tensor(buf134, (1024, 2048), (2048, 1), 0), buf139, buf140, buf142, buf143, buf154, buf156, buf160, buf161, buf163, buf164, buf169, reinterpret_tensor(buf168, (1024, 2048), (2048, 1), 0), buf173, buf174, buf176, buf177, buf188, buf190, buf194, buf195, buf197, buf198, buf203, reinterpret_tensor(buf202, (1024, 2048), (2048, 1), 0), buf207, buf208, buf210, buf214, primals_135, buf217, buf218, buf220, buf221, buf226, buf234, buf236, buf240, buf241, buf243, buf244, reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), buf255, buf257, buf261, buf262, buf264, buf265, buf270, reinterpret_tensor(buf269, (1024, 2048), (2048, 1), 0), buf274, buf275, buf277, buf278, buf290, buf292, buf296, buf297, buf299, buf300, buf311, buf313, buf317, buf318, buf320, buf321, buf326, reinterpret_tensor(buf325, (1024, 2048), (2048, 1), 0), buf330, buf331, buf333, buf334, buf346, buf348, buf352, buf353, buf355, buf356, buf367, buf369, buf373, buf374, buf376, buf377, buf382, reinterpret_tensor(buf381, (1024, 2048), (2048, 1), 0), buf386, buf387, buf389, buf390, buf402, buf404, buf408, buf409, buf411, buf412, buf423, buf425, buf429, buf430, buf432, buf433, buf438, reinterpret_tensor(buf437, (1024, 2048), (2048, 1), 0), buf442, buf443, buf445, buf446, buf458, buf460, buf464, buf465, buf467, buf468, buf479, buf481, buf485, buf486, buf488, buf489, buf494, reinterpret_tensor(buf493, (1024, 2048), (2048, 1), 0), buf498, buf499, buf501, buf502, buf514, buf516, buf520, buf521, buf523, buf524, buf535, buf537, buf541, buf542, buf544, buf545, buf550, reinterpret_tensor(buf549, (1024, 2048), (2048, 1), 0), buf554, buf555, buf557, buf561, buf562, buf566, buf568, reinterpret_tensor(primals_132, (32128, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 2048), (2048, 1), 0), buf570, reinterpret_tensor(primals_130, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 512), (512, 1), 0), reinterpret_tensor(buf534, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf527, (8, 64, 1024), (64, 1, 512), 0), buf532, reinterpret_tensor(buf525, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf526, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 512), (512, 1), 0), reinterpret_tensor(primals_125, (512, 512), (512, 1), 0), reinterpret_tensor(buf513, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf505, (8, 64, 1024), (64, 1, 512), 0), buf511, reinterpret_tensor(buf503, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf504, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_124, (512, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 512), (512, 1), 0), reinterpret_tensor(primals_122, (512, 512), (512, 1), 0), reinterpret_tensor(primals_121, (512, 2048), (2048, 1), 0), buf571, reinterpret_tensor(primals_120, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 512), (512, 1), 0), reinterpret_tensor(buf478, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf471, (8, 64, 1024), (64, 1, 512), 0), buf476, reinterpret_tensor(buf469, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf470, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 512), (512, 1), 0), reinterpret_tensor(buf457, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf449, (8, 64, 1024), (64, 1, 512), 0), buf455, reinterpret_tensor(buf447, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf448, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_114, (512, 512), (512, 1), 0), reinterpret_tensor(primals_113, (512, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 512), (512, 1), 0), reinterpret_tensor(primals_111, (512, 2048), (2048, 1), 0), buf572, reinterpret_tensor(primals_110, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), reinterpret_tensor(buf422, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf415, (8, 64, 1024), (64, 1, 512), 0), buf420, reinterpret_tensor(buf413, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf414, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (512, 1), 0), reinterpret_tensor(primals_106, (512, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 512), (512, 1), 0), reinterpret_tensor(buf401, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf393, (8, 64, 1024), (64, 1, 512), 0), buf399, reinterpret_tensor(buf391, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf392, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_104, (512, 512), (512, 1), 0), reinterpret_tensor(primals_103, (512, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 512), (512, 1), 0), reinterpret_tensor(primals_101, (512, 2048), (2048, 1), 0), buf573, reinterpret_tensor(primals_100, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_99, (512, 512), (512, 1), 0), reinterpret_tensor(buf366, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf359, (8, 64, 1024), (64, 1, 512), 0), buf364, reinterpret_tensor(buf357, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf358, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 512), (512, 1), 0), reinterpret_tensor(primals_95, (512, 512), (512, 1), 0), reinterpret_tensor(buf345, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf337, (8, 64, 1024), (64, 1, 512), 0), buf343, reinterpret_tensor(buf335, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf336, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_94, (512, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 512), (512, 1), 0), reinterpret_tensor(primals_92, (512, 512), (512, 1), 0), reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), buf574, reinterpret_tensor(primals_90, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_89, (512, 512), (512, 1), 0), reinterpret_tensor(buf310, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf303, (8, 64, 1024), (64, 1, 512), 0), buf308, reinterpret_tensor(buf301, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf302, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), reinterpret_tensor(buf289, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf281, (8, 64, 1024), (64, 1, 512), 0), buf287, reinterpret_tensor(buf279, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf280, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_84, (512, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 512), (512, 1), 0), reinterpret_tensor(primals_82, (512, 512), (512, 1), 0), reinterpret_tensor(primals_81, (512, 2048), (2048, 1), 0), buf575, reinterpret_tensor(primals_80, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 512), (512, 1), 0), reinterpret_tensor(buf254, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf247, (8, 64, 1024), (64, 1, 512), 0), buf252, reinterpret_tensor(buf245, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf246, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 512), (512, 1), 0), reinterpret_tensor(primals_75, (512, 512), (512, 1), 0), reinterpret_tensor(buf233, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf224, (8, 64, 1024), (64, 1, 512), 0), buf231, reinterpret_tensor(buf222, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf223, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 512), (512, 1), 0), reinterpret_tensor(primals_71, (512, 512), (512, 1), 0), reinterpret_tensor(primals_70, (512, 2048), (2048, 1), 0), buf576, reinterpret_tensor(primals_69, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_68, (512, 512), (512, 1), 0), reinterpret_tensor(buf187, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf180, (8, 64, 1024), (64, 1, 512), 0), buf185, reinterpret_tensor(buf178, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf179, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_67, (512, 512), (512, 1), 0), reinterpret_tensor(primals_66, (512, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 512), (512, 1), 0), reinterpret_tensor(primals_64, (512, 2048), (2048, 1), 0), buf577, reinterpret_tensor(primals_63, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 512), (512, 1), 0), reinterpret_tensor(buf153, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf146, (8, 64, 1024), (64, 1, 512), 0), buf151, reinterpret_tensor(buf144, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf145, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), reinterpret_tensor(primals_60, (512, 512), (512, 1), 0), reinterpret_tensor(primals_59, (512, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 2048), (2048, 1), 0), buf578, reinterpret_tensor(primals_57, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_56, (512, 512), (512, 1), 0), reinterpret_tensor(buf119, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf112, (8, 64, 1024), (64, 1, 512), 0), buf117, reinterpret_tensor(buf110, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf111, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_55, (512, 512), (512, 1), 0), reinterpret_tensor(primals_54, (512, 512), (512, 1), 0), reinterpret_tensor(primals_53, (512, 512), (512, 1), 0), reinterpret_tensor(primals_52, (512, 2048), (2048, 1), 0), buf579, reinterpret_tensor(primals_51, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_50, (512, 512), (512, 1), 0), reinterpret_tensor(buf85, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf78, (8, 64, 1024), (64, 1, 512), 0), buf83, reinterpret_tensor(buf76, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf77, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 512), (512, 1), 0), reinterpret_tensor(primals_47, (512, 512), (512, 1), 0), reinterpret_tensor(primals_46, (512, 2048), (2048, 1), 0), buf580, reinterpret_tensor(primals_45, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_44, (512, 512), (512, 1), 0), reinterpret_tensor(buf51, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf44, (8, 64, 1024), (64, 1, 512), 0), buf49, reinterpret_tensor(buf42, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf43, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_43, (512, 512), (512, 1), 0), reinterpret_tensor(primals_42, (512, 512), (512, 1), 0), reinterpret_tensor(primals_41, (512, 512), (512, 1), 0), reinterpret_tensor(primals_40, (512, 2048), (2048, 1), 0), buf581, reinterpret_tensor(primals_39, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_38, (512, 512), (512, 1), 0), reinterpret_tensor(buf17, (8, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf9, (8, 64, 1024), (64, 1, 512), 0), buf15, reinterpret_tensor(buf7, (8, 64, 1024), (64, 1, 512), 0), reinterpret_tensor(buf8, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(primals_36, (512, 512), (512, 1), 0), reinterpret_tensor(primals_35, (512, 512), (512, 1), 0), reinterpret_tensor(primals_34, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((32, 8), (8, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((32, 8), (8, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    primals_134 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    primals_135 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5ForConditionalGeneration', benchmark_compiled_module)
