
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


cpp_fused_add_embedding_mean_mul_pow_rsqrt_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp5 = tmp4 * tmp4;
                        tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_abs_add_clone_div_full_like_gt_log_lt_minimum_mul_sub_where_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr1[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr1[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                                auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                                auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp30);
                            }
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>(x3 + ((-1L)*x2));
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
                            auto tmp27 = in_ptr1[static_cast<long>(x1 + (8L*tmp26))];
                            auto tmp28 = static_cast<float>(0.0);
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = decltype(tmp0)(tmp0 + tmp29);
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = std::exp(tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    auto out_ptr2 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp5 = tmp4 * tmp4;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_clone_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr1[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr1[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_relu_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr1[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    tmp11.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                                auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                                auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                                auto tmp22 = c10::convert<long>(x3);
                                auto tmp23 = c10::convert<long>(x2);
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
                            out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (1024L*x1) + (8192L*x0))];
                            auto tmp1 = c10::convert<long>((-1L)*(std::min(0L, x3 + ((-1L)*x2))));
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
                            auto tmp21 = in_ptr1[static_cast<long>(x1 + (8L*tmp20))];
                            auto tmp22 = c10::convert<long>(x3);
                            auto tmp23 = c10::convert<long>(x2);
                            auto tmp24 = tmp22 <= tmp23;
                            auto tmp25 = c10::convert<float>(tmp24);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp25);
                            auto tmp28 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            auto tmp30 = decltype(tmp21)(tmp21 + tmp29);
                            auto tmp31 = decltype(tmp0)(tmp0 + tmp30);
                            auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                            in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (1048576L*x1) + (8388608L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0.exp();
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_clone_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_view_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr0[static_cast<long>(x0)] = tmp1;
                out_ptr1[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp10 = static_cast<float>(0.04419417382415922);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_detach_relu_threshold_backward_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4,
                       bool* out_ptr5,
                       bool* out_ptr6,
                       bool* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr1[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr7[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr2[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr10[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr3[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr13[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr4[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr15[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr5[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr10 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr17[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr6[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr18[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr19[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr7[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr20[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr12 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr21[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr8[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr22[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr13 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr23[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr9[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr24[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134 = args
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
    assert_size_stride(primals_133, (4, 1024), (1024, 1))
    assert_size_stride(primals_134, (4, 1024), (1024, 1))
    buf0 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf2 = reinterpret_tensor(buf1, (4, 1024, 1), (1024, 1, 1), 0); del buf1  # reuse
    buf3 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_view_0(c_void_p(buf2.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf3.data_ptr()))
    buf4 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, reinterpret_tensor(primals_34, (512, 512), (1, 512), 0), out=buf4)
    buf5 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, reinterpret_tensor(primals_35, (512, 512), (1, 512), 0), out=buf5)
    buf6 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, reinterpret_tensor(primals_36, (512, 512), (1, 512), 0), out=buf6)
    buf7 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf8 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    buf9 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf8, (32, 64, 1024), (65536, 1024, 1), 0), out=buf9)
    buf10 = empty((1024, 1024), device='cpu', dtype=torch.int64)
    buf11 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf9, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf9  # reuse
    buf13 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf14 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf5, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf5  # reuse
    cpp_fused__softmax__to_copy_abs_add_clone_div_full_like_gt_log_lt_minimum_mul_sub_where_2(c_void_p(buf12.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf6, (32, 1024, 64), (65536, 64, 1), 0); del buf6  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf14, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf15, (32, 1024, 64), (65536, 64, 1), 0), out=buf16)
    buf17 = buf4; del buf4  # reuse
    cpp_fused_view_3(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = reinterpret_tensor(buf16, (4096, 512), (512, 1), 0); del buf16  # reuse
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf17, reinterpret_tensor(primals_38, (512, 512), (1, 512), 0), out=buf18)
    buf19 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf19, (4, 1024, 1), (1024, 1, 1), 0); del buf19  # reuse
    buf21 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_4(c_void_p(buf20.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf21, reinterpret_tensor(primals_39, (512, 2048), (1, 512), 0), out=buf22)
    buf23 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_5(c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    buf24 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf23, reinterpret_tensor(primals_40, (2048, 512), (1, 2048), 0), out=buf24)
    buf25 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf25, (4, 1024, 1), (1024, 1, 1), 0); del buf25  # reuse
    buf27 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_6(c_void_p(buf26.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_41, (512, 512), (1, 512), 0), out=buf28)
    buf29 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_42, (512, 512), (1, 512), 0), out=buf29)
    buf30 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_43, (512, 512), (1, 512), 0), out=buf30)
    buf31 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf32 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_7(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf32, (32, 64, 1024), (65536, 1024, 1), 0), out=buf33)
    buf34 = buf11; del buf11  # reuse
    buf35 = reinterpret_tensor(buf33, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf33  # reuse
    buf36 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf37 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf38 = reinterpret_tensor(buf29, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf29  # reuse
    cpp_fused__softmax_clone_8(c_void_p(buf35.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf30, (32, 1024, 64), (65536, 64, 1), 0); del buf30  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf37, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf38, (32, 1024, 64), (65536, 64, 1), 0), out=buf39)
    buf40 = buf28; del buf28  # reuse
    cpp_fused_view_9(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf39, (4096, 512), (512, 1), 0); del buf39  # reuse
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, reinterpret_tensor(primals_44, (512, 512), (1, 512), 0), out=buf41)
    buf42 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf43 = reinterpret_tensor(buf42, (4, 1024, 1), (1024, 1, 1), 0); del buf42  # reuse
    buf44 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_10(c_void_p(buf43.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf44, reinterpret_tensor(primals_45, (512, 2048), (1, 512), 0), out=buf45)
    buf46 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_11(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf46, reinterpret_tensor(primals_46, (2048, 512), (1, 2048), 0), out=buf47)
    buf48 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf50 = reinterpret_tensor(buf49, (4, 1024, 1), (1024, 1, 1), 0); del buf49  # reuse
    buf51 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_12(c_void_p(buf50.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf51.data_ptr()))
    buf52 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf51, reinterpret_tensor(primals_47, (512, 512), (1, 512), 0), out=buf52)
    buf53 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf51, reinterpret_tensor(primals_48, (512, 512), (1, 512), 0), out=buf53)
    buf54 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf51, reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), out=buf54)
    buf55 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf56 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_13(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf56, (32, 64, 1024), (65536, 1024, 1), 0), out=buf57)
    buf58 = buf34; del buf34  # reuse
    buf59 = reinterpret_tensor(buf57, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf57  # reuse
    buf60 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf61 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf53, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf53  # reuse
    cpp_fused__softmax_clone_14(c_void_p(buf59.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf54, (32, 1024, 64), (65536, 64, 1), 0); del buf54  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf61, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf62, (32, 1024, 64), (65536, 64, 1), 0), out=buf63)
    buf64 = buf52; del buf52  # reuse
    cpp_fused_view_15(c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = reinterpret_tensor(buf63, (4096, 512), (512, 1), 0); del buf63  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf64, reinterpret_tensor(primals_50, (512, 512), (1, 512), 0), out=buf65)
    buf66 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf66, (4, 1024, 1), (1024, 1, 1), 0); del buf66  # reuse
    buf68 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_16(c_void_p(buf67.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf68.data_ptr()))
    buf69 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf68, reinterpret_tensor(primals_51, (512, 2048), (1, 512), 0), out=buf69)
    buf70 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_17(c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, reinterpret_tensor(primals_52, (2048, 512), (1, 2048), 0), out=buf71)
    buf72 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf73 = reinterpret_tensor(buf72, (4, 1024, 1), (1024, 1, 1), 0); del buf72  # reuse
    buf74 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_18(c_void_p(buf73.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf74, reinterpret_tensor(primals_53, (512, 512), (1, 512), 0), out=buf75)
    buf76 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf74, reinterpret_tensor(primals_54, (512, 512), (1, 512), 0), out=buf76)
    buf77 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf74, reinterpret_tensor(primals_55, (512, 512), (1, 512), 0), out=buf77)
    buf78 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf79 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_19(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf78, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf79, (32, 64, 1024), (65536, 1024, 1), 0), out=buf80)
    buf81 = buf58; del buf58  # reuse
    buf82 = reinterpret_tensor(buf80, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf80  # reuse
    buf83 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf84 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf85 = reinterpret_tensor(buf76, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf76  # reuse
    cpp_fused__softmax_clone_20(c_void_p(buf82.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf77, (32, 1024, 64), (65536, 64, 1), 0); del buf77  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf85, (32, 1024, 64), (65536, 64, 1), 0), out=buf86)
    buf87 = buf75; del buf75  # reuse
    cpp_fused_view_21(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf86, (4096, 512), (512, 1), 0); del buf86  # reuse
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf87, reinterpret_tensor(primals_56, (512, 512), (1, 512), 0), out=buf88)
    buf89 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf90 = reinterpret_tensor(buf89, (4, 1024, 1), (1024, 1, 1), 0); del buf89  # reuse
    buf91 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_22(c_void_p(buf90.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf91, reinterpret_tensor(primals_57, (512, 2048), (1, 512), 0), out=buf92)
    buf93 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_23(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf93, reinterpret_tensor(primals_58, (2048, 512), (1, 2048), 0), out=buf94)
    buf95 = buf48; del buf48  # reuse
    buf96 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf97 = reinterpret_tensor(buf96, (4, 1024, 1), (1024, 1, 1), 0); del buf96  # reuse
    buf98 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_24(c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf98, reinterpret_tensor(primals_59, (512, 512), (1, 512), 0), out=buf99)
    buf100 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf98, reinterpret_tensor(primals_60, (512, 512), (1, 512), 0), out=buf100)
    buf101 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf98, reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), out=buf101)
    buf102 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf103 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_25(c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    buf104 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf102, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf103, (32, 64, 1024), (65536, 1024, 1), 0), out=buf104)
    buf105 = buf81; del buf81  # reuse
    buf106 = reinterpret_tensor(buf104, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf104  # reuse
    buf107 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf108 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf99, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf99  # reuse
    cpp_fused__softmax_clone_26(c_void_p(buf106.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf101, (32, 1024, 64), (65536, 64, 1), 0); del buf101  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf109, (32, 1024, 64), (65536, 64, 1), 0), out=buf110)
    buf111 = buf100; del buf100  # reuse
    cpp_fused_view_27(c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf110, (4096, 512), (512, 1), 0); del buf110  # reuse
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf111, reinterpret_tensor(primals_62, (512, 512), (1, 512), 0), out=buf112)
    buf113 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf114 = reinterpret_tensor(buf113, (4, 1024, 1), (1024, 1, 1), 0); del buf113  # reuse
    buf115 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_28(c_void_p(buf114.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
    extern_kernels.mm(buf115, reinterpret_tensor(primals_63, (512, 2048), (1, 512), 0), out=buf116)
    buf117 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_29(c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, reinterpret_tensor(primals_64, (2048, 512), (1, 2048), 0), out=buf118)
    buf119 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf120 = reinterpret_tensor(buf119, (4, 1024, 1), (1024, 1, 1), 0); del buf119  # reuse
    buf121 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_30(c_void_p(buf120.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, reinterpret_tensor(primals_65, (512, 512), (1, 512), 0), out=buf122)
    buf123 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, reinterpret_tensor(primals_66, (512, 512), (1, 512), 0), out=buf123)
    buf124 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, reinterpret_tensor(primals_67, (512, 512), (1, 512), 0), out=buf124)
    buf125 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf126 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_31(c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    buf127 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf125, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf126, (32, 64, 1024), (65536, 1024, 1), 0), out=buf127)
    buf128 = buf105; del buf105  # reuse
    buf129 = reinterpret_tensor(buf127, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf127  # reuse
    buf130 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf131 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf123, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf123  # reuse
    cpp_fused__softmax_clone_32(c_void_p(buf129.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_37
    buf133 = reinterpret_tensor(buf124, (32, 1024, 64), (65536, 64, 1), 0); del buf124  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf132, (32, 1024, 64), (65536, 64, 1), 0), out=buf133)
    buf134 = buf122; del buf122  # reuse
    cpp_fused_view_33(c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    buf135 = reinterpret_tensor(buf133, (4096, 512), (512, 1), 0); del buf133  # reuse
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf134, reinterpret_tensor(primals_68, (512, 512), (1, 512), 0), out=buf135)
    buf136 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf137 = reinterpret_tensor(buf136, (4, 1024, 1), (1024, 1, 1), 0); del buf136  # reuse
    buf138 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_34(c_void_p(buf137.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf138.data_ptr()))
    buf139 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
    extern_kernels.mm(buf138, reinterpret_tensor(primals_69, (512, 2048), (1, 512), 0), out=buf139)
    buf140 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_35(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    buf141 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf140, reinterpret_tensor(primals_70, (2048, 512), (1, 2048), 0), out=buf141)
    buf142 = buf95; del buf95  # reuse
    buf143 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf144 = reinterpret_tensor(buf143, (4, 1024, 1), (1024, 1, 1), 0); del buf143  # reuse
    buf145 = buf142; del buf142  # reuse
    buf146 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf147, (4, 1024, 1), (1024, 1, 1), 0); del buf147  # reuse
    buf149 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_view_36(c_void_p(buf145.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf149.data_ptr()))
    del primals_33
    buf150 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf149, reinterpret_tensor(primals_71, (512, 512), (1, 512), 0), out=buf150)
    buf151 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf149, reinterpret_tensor(primals_72, (512, 512), (1, 512), 0), out=buf151)
    buf152 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf149, reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), out=buf152)
    buf153 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf154 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_37(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf153, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf154, (32, 64, 1024), (65536, 1024, 1), 0), out=buf155)
    buf156 = empty((1024, 1024), device='cpu', dtype=torch.int64)
    buf157 = buf128; del buf128  # reuse
    buf158 = reinterpret_tensor(buf155, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf155  # reuse
    buf159 = buf158; del buf158  # reuse
    buf160 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf161 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf162 = reinterpret_tensor(buf150, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf150  # reuse
    cpp_fused__softmax__to_copy_add_clone_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_38(c_void_p(buf159.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    buf163 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf161, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf162, (32, 1024, 64), (65536, 64, 1), 0), out=buf163)
    buf164 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_39(c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf163, (4096, 512), (512, 1), 0); del buf163  # reuse
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf164, reinterpret_tensor(primals_75, (512, 512), (1, 512), 0), out=buf165)
    buf166 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf166, (4, 1024, 1), (1024, 1, 1), 0); del buf166  # reuse
    buf168 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_40(c_void_p(buf167.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf168, reinterpret_tensor(primals_76, (512, 512), (1, 512), 0), out=buf169)
    buf170 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (1, 512), 0), out=buf170)
    buf171 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_78, (512, 512), (1, 512), 0), out=buf171)
    buf172 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf173 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_41(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    buf174 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf173, (32, 64, 1024), (65536, 1024, 1), 0), out=buf174)
    buf175 = buf157; del buf157  # reuse
    buf176 = reinterpret_tensor(buf174, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf174  # reuse
    buf177 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf178 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf169, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf169  # reuse
    cpp_fused__softmax_clone_42(c_void_p(buf176.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf178, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf179, (32, 1024, 64), (65536, 64, 1), 0), out=buf180)
    buf181 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_43(c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    buf182 = reinterpret_tensor(buf180, (4096, 512), (512, 1), 0); del buf180  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf181, reinterpret_tensor(primals_79, (512, 512), (1, 512), 0), out=buf182)
    buf183 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf184 = reinterpret_tensor(buf183, (4, 1024, 1), (1024, 1, 1), 0); del buf183  # reuse
    buf185 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_44(c_void_p(buf184.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
    extern_kernels.mm(buf185, reinterpret_tensor(primals_80, (512, 2048), (1, 512), 0), out=buf186)
    buf187 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_45(c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_81, (2048, 512), (1, 2048), 0), out=buf188)
    buf189 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf189, (4, 1024, 1), (1024, 1, 1), 0); del buf189  # reuse
    buf191 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_46(c_void_p(buf190.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf191, reinterpret_tensor(primals_82, (512, 512), (1, 512), 0), out=buf192)
    buf193 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf191, reinterpret_tensor(primals_83, (512, 512), (1, 512), 0), out=buf193)
    buf194 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf191, reinterpret_tensor(primals_84, (512, 512), (1, 512), 0), out=buf194)
    buf195 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf196 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_47(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    buf197 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf196, (32, 64, 1024), (65536, 1024, 1), 0), out=buf197)
    buf198 = buf175; del buf175  # reuse
    buf199 = reinterpret_tensor(buf197, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf197  # reuse
    buf200 = buf199; del buf199  # reuse
    buf201 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf202 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf203 = reinterpret_tensor(buf192, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf192  # reuse
    cpp_fused__softmax_clone_48(c_void_p(buf200.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf202, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf203, (32, 1024, 64), (65536, 64, 1), 0), out=buf204)
    buf205 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_49(c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    buf206 = reinterpret_tensor(buf204, (4096, 512), (512, 1), 0); del buf204  # reuse
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf205, reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), out=buf206)
    buf207 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf209 = reinterpret_tensor(buf208, (4, 1024, 1), (1024, 1, 1), 0); del buf208  # reuse
    buf210 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_50(c_void_p(buf209.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf210, reinterpret_tensor(primals_86, (512, 512), (1, 512), 0), out=buf211)
    buf212 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (1, 512), 0), out=buf212)
    buf213 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_88, (512, 512), (1, 512), 0), out=buf213)
    buf214 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf215 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_51(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf214, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf215, (32, 64, 1024), (65536, 1024, 1), 0), out=buf216)
    buf217 = buf198; del buf198  # reuse
    buf218 = reinterpret_tensor(buf216, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf216  # reuse
    buf219 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf220 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf211, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf211  # reuse
    cpp_fused__softmax_clone_52(c_void_p(buf218.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf221, (32, 1024, 64), (65536, 64, 1), 0), out=buf222)
    buf223 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_53(c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf222, (4096, 512), (512, 1), 0); del buf222  # reuse
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(buf223, reinterpret_tensor(primals_89, (512, 512), (1, 512), 0), out=buf224)
    buf225 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf225, (4, 1024, 1), (1024, 1, 1), 0); del buf225  # reuse
    buf227 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_54(c_void_p(buf226.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf227.data_ptr()))
    buf228 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
    extern_kernels.mm(buf227, reinterpret_tensor(primals_90, (512, 2048), (1, 512), 0), out=buf228)
    buf229 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_55(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    buf230 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf229, reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), out=buf230)
    buf231 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf232 = reinterpret_tensor(buf231, (4, 1024, 1), (1024, 1, 1), 0); del buf231  # reuse
    buf233 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_56(c_void_p(buf232.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf233, reinterpret_tensor(primals_92, (512, 512), (1, 512), 0), out=buf234)
    buf235 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf233, reinterpret_tensor(primals_93, (512, 512), (1, 512), 0), out=buf235)
    buf236 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf233, reinterpret_tensor(primals_94, (512, 512), (1, 512), 0), out=buf236)
    buf237 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf238 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_57(c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf237, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf238, (32, 64, 1024), (65536, 1024, 1), 0), out=buf239)
    buf240 = buf217; del buf217  # reuse
    buf241 = reinterpret_tensor(buf239, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf239  # reuse
    buf242 = buf241; del buf241  # reuse
    buf243 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf244 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf245 = reinterpret_tensor(buf234, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf234  # reuse
    cpp_fused__softmax_clone_58(c_void_p(buf242.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf244, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf245, (32, 1024, 64), (65536, 64, 1), 0), out=buf246)
    buf247 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_59(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = reinterpret_tensor(buf246, (4096, 512), (512, 1), 0); del buf246  # reuse
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf247, reinterpret_tensor(primals_95, (512, 512), (1, 512), 0), out=buf248)
    buf249 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf250 = reinterpret_tensor(buf249, (4, 1024, 1), (1024, 1, 1), 0); del buf249  # reuse
    buf251 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_60(c_void_p(buf250.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf251, reinterpret_tensor(primals_96, (512, 512), (1, 512), 0), out=buf252)
    buf253 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), out=buf253)
    buf254 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_98, (512, 512), (1, 512), 0), out=buf254)
    buf255 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf256 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_61(c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf255, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf256, (32, 64, 1024), (65536, 1024, 1), 0), out=buf257)
    buf258 = buf240; del buf240  # reuse
    buf259 = reinterpret_tensor(buf257, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf257  # reuse
    buf260 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf261 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf262 = reinterpret_tensor(buf252, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf252  # reuse
    cpp_fused__softmax_clone_62(c_void_p(buf259.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf262, (32, 1024, 64), (65536, 64, 1), 0), out=buf263)
    buf264 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_63(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf263, (4096, 512), (512, 1), 0); del buf263  # reuse
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(buf264, reinterpret_tensor(primals_99, (512, 512), (1, 512), 0), out=buf265)
    buf266 = buf207; del buf207  # reuse
    buf267 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf268 = reinterpret_tensor(buf267, (4, 1024, 1), (1024, 1, 1), 0); del buf267  # reuse
    buf269 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_64(c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
    extern_kernels.mm(buf269, reinterpret_tensor(primals_100, (512, 2048), (1, 512), 0), out=buf270)
    buf271 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_65(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    buf272 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf271, reinterpret_tensor(primals_101, (2048, 512), (1, 2048), 0), out=buf272)
    buf273 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf274 = reinterpret_tensor(buf273, (4, 1024, 1), (1024, 1, 1), 0); del buf273  # reuse
    buf275 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_66(c_void_p(buf274.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf275.data_ptr()))
    buf276 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_102, (512, 512), (1, 512), 0), out=buf276)
    buf277 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_103, (512, 512), (1, 512), 0), out=buf277)
    buf278 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_104, (512, 512), (1, 512), 0), out=buf278)
    buf279 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf280 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_67(c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf280, (32, 64, 1024), (65536, 1024, 1), 0), out=buf281)
    buf282 = buf258; del buf258  # reuse
    buf283 = reinterpret_tensor(buf281, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf281  # reuse
    buf284 = buf283; del buf283  # reuse
    buf285 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf286 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf287 = reinterpret_tensor(buf276, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf276  # reuse
    cpp_fused__softmax_clone_68(c_void_p(buf284.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf286, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf287, (32, 1024, 64), (65536, 64, 1), 0), out=buf288)
    buf289 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_69(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf288, (4096, 512), (512, 1), 0); del buf288  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(buf289, reinterpret_tensor(primals_105, (512, 512), (1, 512), 0), out=buf290)
    buf291 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf292 = reinterpret_tensor(buf291, (4, 1024, 1), (1024, 1, 1), 0); del buf291  # reuse
    buf293 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_70(c_void_p(buf292.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf293, reinterpret_tensor(primals_106, (512, 512), (1, 512), 0), out=buf294)
    buf295 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (1, 512), 0), out=buf295)
    buf296 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 512), (1, 512), 0), out=buf296)
    buf297 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf298 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_71(c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    buf299 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf297, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf298, (32, 64, 1024), (65536, 1024, 1), 0), out=buf299)
    buf300 = buf282; del buf282  # reuse
    buf301 = reinterpret_tensor(buf299, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf299  # reuse
    buf302 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf303 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf304 = reinterpret_tensor(buf294, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf294  # reuse
    cpp_fused__softmax_clone_72(c_void_p(buf301.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf303, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf304, (32, 1024, 64), (65536, 64, 1), 0), out=buf305)
    buf306 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_73(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    buf307 = reinterpret_tensor(buf305, (4096, 512), (512, 1), 0); del buf305  # reuse
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf306, reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), out=buf307)
    buf308 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf309 = reinterpret_tensor(buf308, (4, 1024, 1), (1024, 1, 1), 0); del buf308  # reuse
    buf310 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_74(c_void_p(buf309.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf310.data_ptr()))
    buf311 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
    extern_kernels.mm(buf310, reinterpret_tensor(primals_110, (512, 2048), (1, 512), 0), out=buf311)
    buf312 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_75(c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    buf313 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(buf312, reinterpret_tensor(primals_111, (2048, 512), (1, 2048), 0), out=buf313)
    buf314 = buf266; del buf266  # reuse
    buf315 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf316 = reinterpret_tensor(buf315, (4, 1024, 1), (1024, 1, 1), 0); del buf315  # reuse
    buf317 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_76(c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf317, reinterpret_tensor(primals_112, (512, 512), (1, 512), 0), out=buf318)
    buf319 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf317, reinterpret_tensor(primals_113, (512, 512), (1, 512), 0), out=buf319)
    buf320 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf317, reinterpret_tensor(primals_114, (512, 512), (1, 512), 0), out=buf320)
    buf321 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf322 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_77(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    buf323 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf321, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf322, (32, 64, 1024), (65536, 1024, 1), 0), out=buf323)
    buf324 = buf300; del buf300  # reuse
    buf325 = reinterpret_tensor(buf323, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf323  # reuse
    buf326 = buf325; del buf325  # reuse
    buf327 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf328 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf329 = reinterpret_tensor(buf318, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf318  # reuse
    cpp_fused__softmax_clone_78(c_void_p(buf326.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    buf330 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf328, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf329, (32, 1024, 64), (65536, 64, 1), 0), out=buf330)
    buf331 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_79(c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    buf332 = reinterpret_tensor(buf330, (4096, 512), (512, 1), 0); del buf330  # reuse
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf331, reinterpret_tensor(primals_115, (512, 512), (1, 512), 0), out=buf332)
    buf333 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf334 = reinterpret_tensor(buf333, (4, 1024, 1), (1024, 1, 1), 0); del buf333  # reuse
    buf335 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_80(c_void_p(buf334.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf335, reinterpret_tensor(primals_116, (512, 512), (1, 512), 0), out=buf336)
    buf337 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (1, 512), 0), out=buf337)
    buf338 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 512), (1, 512), 0), out=buf338)
    buf339 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf340 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_81(c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf340, (32, 64, 1024), (65536, 1024, 1), 0), out=buf341)
    buf342 = buf324; del buf324  # reuse
    buf343 = reinterpret_tensor(buf341, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf341  # reuse
    buf344 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf345 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf406 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf346 = reinterpret_tensor(buf336, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf336  # reuse
    cpp_fused__softmax_clone_detach_82(c_void_p(buf343.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf346.data_ptr()))
    buf347 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf345, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf346, (32, 1024, 64), (65536, 64, 1), 0), out=buf347)
    buf348 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_83(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = reinterpret_tensor(buf347, (4096, 512), (512, 1), 0); del buf347  # reuse
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(buf348, reinterpret_tensor(primals_119, (512, 512), (1, 512), 0), out=buf349)
    buf350 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf350, (4, 1024, 1), (1024, 1, 1), 0); del buf350  # reuse
    buf352 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_84(c_void_p(buf351.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
    extern_kernels.mm(buf352, reinterpret_tensor(primals_120, (512, 2048), (1, 512), 0), out=buf353)
    buf354 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    buf405 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_view_85(c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf405.data_ptr()))
    buf355 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf354, reinterpret_tensor(primals_121, (2048, 512), (1, 2048), 0), out=buf355)
    buf356 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf357 = reinterpret_tensor(buf356, (4, 1024, 1), (1024, 1, 1), 0); del buf356  # reuse
    buf358 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_86(c_void_p(buf357.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf358.data_ptr()))
    buf359 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf358, reinterpret_tensor(primals_122, (512, 512), (1, 512), 0), out=buf359)
    buf360 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf358, reinterpret_tensor(primals_123, (512, 512), (1, 512), 0), out=buf360)
    buf361 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf358, reinterpret_tensor(primals_124, (512, 512), (1, 512), 0), out=buf361)
    buf362 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf363 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_87(c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf343, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf343  # reuse
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf362, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf363, (32, 64, 1024), (65536, 1024, 1), 0), out=buf364)
    buf365 = buf344; del buf344  # reuse
    buf366 = reinterpret_tensor(buf364, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf364  # reuse
    buf367 = buf366; del buf366  # reuse
    buf368 = buf342; del buf342  # reuse
    buf369 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf404 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf370 = reinterpret_tensor(buf359, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf359  # reuse
    cpp_fused__softmax_clone_detach_88(c_void_p(buf367.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf370.data_ptr()))
    del primals_74
    buf371 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf369, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf370, (32, 1024, 64), (65536, 64, 1), 0), out=buf371)
    buf372 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_89(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    buf373 = reinterpret_tensor(buf371, (4096, 512), (512, 1), 0); del buf371  # reuse
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf372, reinterpret_tensor(primals_125, (512, 512), (1, 512), 0), out=buf373)
    buf374 = buf314; del buf314  # reuse
    buf375 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf376 = reinterpret_tensor(buf375, (4, 1024, 1), (1024, 1, 1), 0); del buf375  # reuse
    buf377 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_90(c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf377, reinterpret_tensor(primals_126, (512, 512), (1, 512), 0), out=buf378)
    buf379 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (1, 512), 0), out=buf379)
    buf380 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), reinterpret_tensor(primals_128, (512, 512), (1, 512), 0), out=buf380)
    buf381 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    buf382 = empty((4, 8, 64, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_91(c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    buf383 = reinterpret_tensor(buf367, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf367  # reuse
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf381, (32, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf382, (32, 64, 1024), (65536, 1024, 1), 0), out=buf383)
    buf384 = buf368; del buf368  # reuse
    buf385 = reinterpret_tensor(buf383, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf383  # reuse
    buf386 = buf365; del buf365  # reuse
    buf387 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf403 = empty((4, 8, 1024, 1024), device='cpu', dtype=torch.float32)
    buf388 = reinterpret_tensor(buf378, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf378  # reuse
    cpp_fused__softmax_clone_detach_92(c_void_p(buf385.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf388.data_ptr()))
    del buf384
    del buf385
    del buf386
    buf389 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf387, (32, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf388, (32, 1024, 64), (65536, 64, 1), 0), out=buf389)
    buf390 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_93(c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = reinterpret_tensor(buf389, (4096, 512), (512, 1), 0); del buf389  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, reinterpret_tensor(primals_129, (512, 512), (1, 512), 0), out=buf391)
    buf392 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf393 = reinterpret_tensor(buf392, (4, 1024, 1), (1024, 1, 1), 0); del buf392  # reuse
    buf394 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_view_94(c_void_p(buf393.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = buf353; del buf353  # reuse
    # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
    extern_kernels.mm(buf394, reinterpret_tensor(primals_130, (512, 2048), (1, 512), 0), out=buf395)
    buf396 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    buf402 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_view_95(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf402.data_ptr()))
    del buf395
    buf397 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(buf396, reinterpret_tensor(primals_131, (2048, 512), (1, 2048), 0), out=buf397)
    buf398 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf399 = reinterpret_tensor(buf398, (4, 1024, 1), (1024, 1, 1), 0); del buf398  # reuse
    buf400 = reinterpret_tensor(buf374, (4096, 512), (512, 1), 0); del buf374  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_96(c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(primals_32.data_ptr()))
    buf401 = empty((4096, 32128), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(buf400, reinterpret_tensor(primals_132, (512, 32128), (1, 512), 0), out=buf401)
    buf407 = buf326; del buf326  # reuse
    buf408 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf409 = buf301; del buf301  # reuse
    buf410 = buf284; del buf284  # reuse
    buf411 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf412 = buf259; del buf259  # reuse
    buf413 = buf242; del buf242  # reuse
    buf414 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf415 = buf218; del buf218  # reuse
    buf416 = buf200; del buf200  # reuse
    buf417 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf418 = buf176; del buf176  # reuse
    buf419 = buf159; del buf159  # reuse
    buf420 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf421 = buf129; del buf129  # reuse
    buf422 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf423 = buf106; del buf106  # reuse
    buf424 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf425 = buf82; del buf82  # reuse
    buf426 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf427 = buf59; del buf59  # reuse
    buf428 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf429 = buf35; del buf35  # reuse
    buf430 = empty((4, 1024, 2048), device='cpu', dtype=torch.bool)
    buf431 = buf12; del buf12  # reuse
    cpp_fused__softmax_detach_relu_threshold_backward_97(c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()))
    return (reinterpret_tensor(buf401, (4, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf151, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf152, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf170, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf171, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf193, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf194, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf212, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf213, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf235, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf236, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf253, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf254, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf277, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf278, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf295, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf296, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf319, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf320, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf337, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf338, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf360, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf361, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf379, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf380, (4, 8, 1024, 64), (524288, 64, 512, 1), 0), buf145, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_133, buf0, buf2, buf3, buf10, buf17, buf18, buf20, buf21, buf23, buf24, buf26, buf27, buf40, buf41, buf43, buf44, buf46, buf47, buf50, buf51, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf74, buf87, buf88, buf90, buf91, buf93, buf94, buf97, buf98, buf111, buf112, buf114, buf115, buf117, buf118, buf120, buf121, buf134, buf135, buf137, buf138, buf140, buf141, buf144, primals_134, buf146, buf148, buf149, buf156, buf164, buf165, buf167, buf168, reinterpret_tensor(buf145, (4096, 512), (512, 1), 0), buf181, buf182, buf184, buf185, buf187, buf188, buf190, buf191, buf205, buf206, buf209, buf210, buf223, buf224, buf226, buf227, buf229, buf230, buf232, buf233, buf247, buf248, buf250, buf251, buf264, buf265, buf268, buf269, buf271, buf272, buf274, buf275, buf289, buf290, buf292, buf293, buf306, buf307, buf309, buf310, buf312, buf313, buf316, buf317, buf331, buf332, buf334, buf335, buf348, buf349, buf351, buf352, buf354, buf355, buf357, buf358, buf372, buf373, buf376, buf377, buf390, buf391, buf393, buf394, buf396, buf397, buf399, buf400, reinterpret_tensor(primals_132, (32128, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 2048), (2048, 1), 0), buf402, reinterpret_tensor(primals_130, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 512), (512, 1), 0), reinterpret_tensor(buf387, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf388, (32, 64, 1024), (65536, 1, 64), 0), buf403, reinterpret_tensor(buf381, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf382, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_128, (512, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 512), (512, 1), 0), reinterpret_tensor(primals_125, (512, 512), (512, 1), 0), reinterpret_tensor(buf369, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf370, (32, 64, 1024), (65536, 1, 64), 0), buf404, reinterpret_tensor(buf362, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf363, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_124, (512, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 512), (512, 1), 0), reinterpret_tensor(primals_122, (512, 512), (512, 1), 0), reinterpret_tensor(primals_121, (512, 2048), (2048, 1), 0), buf405, reinterpret_tensor(primals_120, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 512), (512, 1), 0), reinterpret_tensor(buf345, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf346, (32, 64, 1024), (65536, 1, 64), 0), buf406, reinterpret_tensor(buf339, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf340, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_118, (512, 512), (512, 1), 0), reinterpret_tensor(primals_117, (512, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 512), (512, 1), 0), reinterpret_tensor(buf328, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf329, (32, 64, 1024), (65536, 1, 64), 0), buf407, reinterpret_tensor(buf321, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf322, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_114, (512, 512), (512, 1), 0), reinterpret_tensor(primals_113, (512, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 512), (512, 1), 0), reinterpret_tensor(primals_111, (512, 2048), (2048, 1), 0), buf408, reinterpret_tensor(primals_110, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), reinterpret_tensor(buf303, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf304, (32, 64, 1024), (65536, 1, 64), 0), buf409, reinterpret_tensor(buf297, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf298, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_108, (512, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 512), (512, 1), 0), reinterpret_tensor(primals_106, (512, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 512), (512, 1), 0), reinterpret_tensor(buf286, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf287, (32, 64, 1024), (65536, 1, 64), 0), buf410, reinterpret_tensor(buf279, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf280, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_104, (512, 512), (512, 1), 0), reinterpret_tensor(primals_103, (512, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 512), (512, 1), 0), reinterpret_tensor(primals_101, (512, 2048), (2048, 1), 0), buf411, reinterpret_tensor(primals_100, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_99, (512, 512), (512, 1), 0), reinterpret_tensor(buf261, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf262, (32, 64, 1024), (65536, 1, 64), 0), buf412, reinterpret_tensor(buf255, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf256, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_98, (512, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 512), (512, 1), 0), reinterpret_tensor(primals_95, (512, 512), (512, 1), 0), reinterpret_tensor(buf244, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf245, (32, 64, 1024), (65536, 1, 64), 0), buf413, reinterpret_tensor(buf237, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf238, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_94, (512, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 512), (512, 1), 0), reinterpret_tensor(primals_92, (512, 512), (512, 1), 0), reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), buf414, reinterpret_tensor(primals_90, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_89, (512, 512), (512, 1), 0), reinterpret_tensor(buf220, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf221, (32, 64, 1024), (65536, 1, 64), 0), buf415, reinterpret_tensor(buf214, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf215, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_88, (512, 512), (512, 1), 0), reinterpret_tensor(primals_87, (512, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), reinterpret_tensor(buf202, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf203, (32, 64, 1024), (65536, 1, 64), 0), buf416, reinterpret_tensor(buf195, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf196, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_84, (512, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 512), (512, 1), 0), reinterpret_tensor(primals_82, (512, 512), (512, 1), 0), reinterpret_tensor(primals_81, (512, 2048), (2048, 1), 0), buf417, reinterpret_tensor(primals_80, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 512), (512, 1), 0), reinterpret_tensor(buf178, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf179, (32, 64, 1024), (65536, 1, 64), 0), buf418, reinterpret_tensor(buf172, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf173, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_78, (512, 512), (512, 1), 0), reinterpret_tensor(primals_77, (512, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 512), (512, 1), 0), reinterpret_tensor(primals_75, (512, 512), (512, 1), 0), reinterpret_tensor(buf161, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf162, (32, 64, 1024), (65536, 1, 64), 0), buf419, reinterpret_tensor(buf153, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf154, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 512), (512, 1), 0), reinterpret_tensor(primals_71, (512, 512), (512, 1), 0), reinterpret_tensor(primals_70, (512, 2048), (2048, 1), 0), buf420, reinterpret_tensor(primals_69, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_68, (512, 512), (512, 1), 0), reinterpret_tensor(buf131, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf132, (32, 64, 1024), (65536, 1, 64), 0), buf421, reinterpret_tensor(buf125, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf126, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_67, (512, 512), (512, 1), 0), reinterpret_tensor(primals_66, (512, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 512), (512, 1), 0), reinterpret_tensor(primals_64, (512, 2048), (2048, 1), 0), buf422, reinterpret_tensor(primals_63, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 512), (512, 1), 0), reinterpret_tensor(buf108, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf109, (32, 64, 1024), (65536, 1, 64), 0), buf423, reinterpret_tensor(buf102, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf103, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), reinterpret_tensor(primals_60, (512, 512), (512, 1), 0), reinterpret_tensor(primals_59, (512, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 2048), (2048, 1), 0), buf424, reinterpret_tensor(primals_57, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_56, (512, 512), (512, 1), 0), reinterpret_tensor(buf84, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf85, (32, 64, 1024), (65536, 1, 64), 0), buf425, reinterpret_tensor(buf78, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf79, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_55, (512, 512), (512, 1), 0), reinterpret_tensor(primals_54, (512, 512), (512, 1), 0), reinterpret_tensor(primals_53, (512, 512), (512, 1), 0), reinterpret_tensor(primals_52, (512, 2048), (2048, 1), 0), buf426, reinterpret_tensor(primals_51, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_50, (512, 512), (512, 1), 0), reinterpret_tensor(buf61, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf62, (32, 64, 1024), (65536, 1, 64), 0), buf427, reinterpret_tensor(buf55, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf56, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 512), (512, 1), 0), reinterpret_tensor(primals_47, (512, 512), (512, 1), 0), reinterpret_tensor(primals_46, (512, 2048), (2048, 1), 0), buf428, reinterpret_tensor(primals_45, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_44, (512, 512), (512, 1), 0), reinterpret_tensor(buf37, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf38, (32, 64, 1024), (65536, 1, 64), 0), buf429, reinterpret_tensor(buf31, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf32, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_43, (512, 512), (512, 1), 0), reinterpret_tensor(primals_42, (512, 512), (512, 1), 0), reinterpret_tensor(primals_41, (512, 512), (512, 1), 0), reinterpret_tensor(primals_40, (512, 2048), (2048, 1), 0), buf430, reinterpret_tensor(primals_39, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_38, (512, 512), (512, 1), 0), reinterpret_tensor(buf14, (32, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf15, (32, 64, 1024), (65536, 1, 64), 0), buf431, reinterpret_tensor(buf7, (32, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf8, (32, 1024, 64), (65536, 1, 1024), 0), reinterpret_tensor(primals_36, (512, 512), (512, 1), 0), reinterpret_tensor(primals_35, (512, 512), (512, 1), 0), reinterpret_tensor(primals_34, (512, 512), (512, 1), 0), )


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
    primals_133 = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    primals_134 = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
