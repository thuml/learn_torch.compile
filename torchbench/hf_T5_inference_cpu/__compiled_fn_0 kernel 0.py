
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


cpp_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp1)(tmp1 + 32128);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 32128L), "index out of bounds: 0 <= tmp4 < 32128L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_2 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        auto tmp9 = decltype(tmp8)(tmp8 + 32128);
                        auto tmp10 = tmp8 < 0;
                        auto tmp11 = tmp10 ? tmp9 : tmp8;
                        TORCH_CHECK((0 <= tmp11) & (tmp11 < 32128L), "index out of bounds: 0 <= tmp11 < 32128L")
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp11)));
                        auto tmp13 = tmp12 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp18 = in_ptr3[static_cast<long>(x0)];
                    auto tmp23 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp1)(tmp1 + 32128);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 32128L), "index out of bounds: 0 <= tmp4 < 32128L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp19 = decltype(tmp18)(tmp18 + 32128);
                    auto tmp20 = tmp18 < 0;
                    auto tmp21 = tmp20 ? tmp19 : tmp18;
                    TORCH_CHECK((0 <= tmp21) & (tmp21 < 32128L), "index out of bounds: 0 <= tmp21 < 32128L")
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp21)));
                    auto tmp24 = tmp23 / tmp9;
                    auto tmp25 = decltype(tmp24)(tmp24 + tmp11);
                    auto tmp26 = 1 / std::sqrt(tmp25);
                    auto tmp27 = at::vec::Vectorized<float>(tmp26);
                    auto tmp28 = tmp22 * tmp27;
                    auto tmp29 = tmp17 * tmp28;
                    tmp16.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp29.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp1)(tmp1 + 32128);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 32128L), "index out of bounds: 0 <= tmp4 < 32128L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = tmp8 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp1)(tmp1 + 32128);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 32128L), "index out of bounds: 0 <= tmp4 < 32128L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp0 * tmp17;
                    tmp18.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_14 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = tmp10 * tmp10;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_20 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_21 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_28 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_35 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp31 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = tmp8 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp1)(tmp1 + 32128);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 32128L), "index out of bounds: 0 <= tmp4 < 32128L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp0 * tmp17;
                    tmp18.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32128);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32128L), "index out of bounds: 0 <= tmp3 < 32128L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = tmp10 * tmp10;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_56 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_68 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_80 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_84 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_92 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(512.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = static_cast<float>(1e-06);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp1 * tmp8;
                    auto tmp10 = tmp0 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_104 = async_compile.cpp('''
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
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
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
                            out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))];
                            auto tmp32 = out_ptr0[static_cast<long>(x2 + (512L*x1) + (4096L*x0))];
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
                            in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (2097152L*x0))] = tmp33;
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (262144L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (262144L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (32768L*x2) + (262144L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (262144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp17 = static_cast<float>(0.04419417382415922);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (512, ), (1, ))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (32128, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 512), (512, 1))
    assert_size_stride(arg34_1, (512, 512), (512, 1))
    assert_size_stride(arg35_1, (512, 512), (512, 1))
    assert_size_stride(arg36_1, (32, 8), (8, 1))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (2048, 512), (512, 1))
    assert_size_stride(arg39_1, (512, 2048), (2048, 1))
    assert_size_stride(arg40_1, (512, 512), (512, 1))
    assert_size_stride(arg41_1, (512, 512), (512, 1))
    assert_size_stride(arg42_1, (512, 512), (512, 1))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (2048, 512), (512, 1))
    assert_size_stride(arg45_1, (512, 2048), (2048, 1))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, 512), (512, 1))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, 512), (512, 1))
    assert_size_stride(arg50_1, (2048, 512), (512, 1))
    assert_size_stride(arg51_1, (512, 2048), (2048, 1))
    assert_size_stride(arg52_1, (512, 512), (512, 1))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (2048, 512), (512, 1))
    assert_size_stride(arg57_1, (512, 2048), (2048, 1))
    assert_size_stride(arg58_1, (512, 512), (512, 1))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 512), (512, 1))
    assert_size_stride(arg62_1, (2048, 512), (512, 1))
    assert_size_stride(arg63_1, (512, 2048), (2048, 1))
    assert_size_stride(arg64_1, (512, 512), (512, 1))
    assert_size_stride(arg65_1, (512, 512), (512, 1))
    assert_size_stride(arg66_1, (512, 512), (512, 1))
    assert_size_stride(arg67_1, (512, 512), (512, 1))
    assert_size_stride(arg68_1, (2048, 512), (512, 1))
    assert_size_stride(arg69_1, (512, 2048), (2048, 1))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (32, 8), (8, 1))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, 512), (512, 1))
    assert_size_stride(arg77_1, (512, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 512), (512, 1))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (512, 2048), (2048, 1))
    assert_size_stride(arg81_1, (512, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 512), (512, 1))
    assert_size_stride(arg83_1, (512, 512), (512, 1))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, 512), (512, 1))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, 512), (512, 1))
    assert_size_stride(arg93_1, (512, 512), (512, 1))
    assert_size_stride(arg94_1, (512, 512), (512, 1))
    assert_size_stride(arg95_1, (512, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, 512), (512, 1))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (2048, 512), (512, 1))
    assert_size_stride(arg100_1, (512, 2048), (2048, 1))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, 512), (512, 1))
    assert_size_stride(arg103_1, (512, 512), (512, 1))
    assert_size_stride(arg104_1, (512, 512), (512, 1))
    assert_size_stride(arg105_1, (512, 512), (512, 1))
    assert_size_stride(arg106_1, (512, 512), (512, 1))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (2048, 512), (512, 1))
    assert_size_stride(arg110_1, (512, 2048), (2048, 1))
    assert_size_stride(arg111_1, (512, 512), (512, 1))
    assert_size_stride(arg112_1, (512, 512), (512, 1))
    assert_size_stride(arg113_1, (512, 512), (512, 1))
    assert_size_stride(arg114_1, (512, 512), (512, 1))
    assert_size_stride(arg115_1, (512, 512), (512, 1))
    assert_size_stride(arg116_1, (512, 512), (512, 1))
    assert_size_stride(arg117_1, (512, 512), (512, 1))
    assert_size_stride(arg118_1, (512, 512), (512, 1))
    assert_size_stride(arg119_1, (2048, 512), (512, 1))
    assert_size_stride(arg120_1, (512, 2048), (2048, 1))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, 512), (512, 1))
    assert_size_stride(arg125_1, (512, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 512), (512, 1))
    assert_size_stride(arg127_1, (512, 512), (512, 1))
    assert_size_stride(arg128_1, (512, 512), (512, 1))
    assert_size_stride(arg129_1, (2048, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 2048), (2048, 1))
    assert_size_stride(arg131_1, (32128, 512), (512, 1))
    assert_size_stride(arg132_1, (4, 512), (512, 1))
    assert_size_stride(arg133_1, (4, 512), (512, 1))
    buf0 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf1 = empty((4, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(c_void_p(arg133_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg13_1
    buf2 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf2)
    del arg70_1
    buf3 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), out=buf3)
    del arg71_1
    buf4 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf5 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    buf6 = empty((32, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf5, (32, 64, 512), (32768, 512, 1), 0), out=buf6)
    buf7 = empty_strided((4, 8, 512, 1), (4096, 512, 1, 16384), device='cpu', dtype=torch.float32)
    buf8 = reinterpret_tensor(buf6, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf6  # reuse
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((4, 8, 512, 1), (4096, 512, 1, 16384), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_2(c_void_p(buf9.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = reinterpret_tensor(buf5, (2048, 512), (512, 1), 0); del buf5  # reuse
    # Source Nodes: [l__mod___model_decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (2048, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf11)
    del arg72_1
    buf12 = buf9; del buf9  # reuse
    buf13 = reinterpret_tensor(buf1, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf1  # reuse
    cpp_fused__softmax_clone_3(c_void_p(buf12.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = reinterpret_tensor(buf4, (32, 512, 64), (32768, 64, 1), 0); del buf4  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf12, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf13, (32, 512, 64), (32768, 64, 1), 0), out=buf14)
    buf15 = reinterpret_tensor(buf13, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf13  # reuse
    cpp_fused_clone_4(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf14, (2048, 512), (512, 1), 0); del buf14  # reuse
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf16)
    del arg74_1
    buf17 = buf0; del buf0  # reuse
    buf20 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf15, (4, 512, 512), (262144, 512, 1), 0); del buf15  # reuse
    buf21 = reinterpret_tensor(buf2, (4, 512, 512), (262144, 512, 1), 0); del buf2  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_5(c_void_p(arg133_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg0_1
    del arg14_1
    del buf17
    buf19 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (2048, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf19)
    del arg75_1
    buf22 = reinterpret_tensor(buf18, (2048, 512), (512, 1), 0); del buf18  # reuse
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (2048, 512), (512, 1), 0), reinterpret_tensor(arg33_1, (512, 512), (1, 512), 0), out=buf22)
    del arg33_1
    buf23 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (2048, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 512), (1, 512), 0), out=buf23)
    del arg34_1
    buf24 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf25 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf12, (32, 512, 512), (262144, 512, 1), 0); del buf12  # reuse
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf25, (32, 64, 512), (32768, 512, 1), 0), out=buf26)
    buf27 = buf10; del buf10  # reuse
    buf28 = reinterpret_tensor(buf26, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf26  # reuse
    buf29 = buf7; del buf7  # reuse
    cpp_fused__softmax_7(c_void_p(buf28.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf25, (2048, 512), (512, 1), 0); del buf25  # reuse
    # Source Nodes: [l__mod___model_encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (2048, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 512), (1, 512), 0), out=buf30)
    del arg35_1
    buf31 = buf28; del buf28  # reuse
    buf32 = reinterpret_tensor(buf21, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf21  # reuse
    cpp_fused__softmax_clone_8(c_void_p(buf31.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf30, (32, 512, 64), (32768, 64, 1), 0); del buf30  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf32, (32, 512, 64), (32768, 64, 1), 0), out=buf33)
    buf34 = reinterpret_tensor(buf32, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf32  # reuse
    cpp_fused_clone_9(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    buf35 = reinterpret_tensor(buf33, (2048, 512), (512, 1), 0); del buf33  # reuse
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (2048, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf35)
    del arg37_1
    buf36 = buf20; del buf20  # reuse
    buf37 = reinterpret_tensor(buf34, (4, 512, 512), (262144, 512, 1), 0); del buf34  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_10(c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg1_1
    buf38 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (2048, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 2048), (1, 512), 0), out=buf38)
    del arg38_1
    buf39 = reinterpret_tensor(buf38, (4, 512, 2048), (1048576, 2048, 1), 0); del buf38  # reuse
    cpp_fused_relu_11(c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf37, (2048, 512), (512, 1), 0); del buf37  # reuse
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 512), (1, 2048), 0), out=buf40)
    del arg39_1
    buf41 = buf36; del buf36  # reuse
    buf42 = reinterpret_tensor(buf24, (4, 512, 512), (262144, 512, 1), 0); del buf24  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_12(c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg2_1
    buf43 = buf23; del buf23  # reuse
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf43)
    del arg40_1
    buf44 = buf22; del buf22  # reuse
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), out=buf44)
    del arg41_1
    buf45 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf46 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_13(c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf31, (32, 512, 512), (262144, 512, 1), 0); del buf31  # reuse
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf45, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf46, (32, 64, 512), (32768, 512, 1), 0), out=buf47)
    buf48 = buf29; del buf29  # reuse
    buf49 = reinterpret_tensor(buf47, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf47  # reuse
    buf50 = buf27; del buf27  # reuse
    cpp_fused__softmax_14(c_void_p(buf49.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf46, (2048, 512), (512, 1), 0); del buf46  # reuse
    # Source Nodes: [l__mod___model_encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (2048, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 512), (1, 512), 0), out=buf51)
    del arg42_1
    buf52 = buf49; del buf49  # reuse
    buf53 = reinterpret_tensor(buf42, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf42  # reuse
    cpp_fused__softmax_clone_15(c_void_p(buf52.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf51, (32, 512, 64), (32768, 64, 1), 0); del buf51  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf52, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf53, (32, 512, 64), (32768, 64, 1), 0), out=buf54)
    buf55 = reinterpret_tensor(buf53, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf53  # reuse
    cpp_fused_clone_16(c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    buf56 = reinterpret_tensor(buf54, (2048, 512), (512, 1), 0); del buf54  # reuse
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (2048, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf56)
    del arg43_1
    buf57 = reinterpret_tensor(buf35, (4, 512, 512), (262144, 512, 1), 0); del buf35  # reuse
    buf58 = buf41; del buf41  # reuse
    buf59 = reinterpret_tensor(buf55, (4, 512, 512), (262144, 512, 1), 0); del buf55  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_17(c_void_p(buf57.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del arg132_1
    del arg3_1
    buf60 = reinterpret_tensor(buf39, (2048, 2048), (2048, 1), 0); del buf39  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf59, (2048, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 2048), (1, 512), 0), out=buf60)
    del arg44_1
    buf61 = reinterpret_tensor(buf60, (4, 512, 2048), (1048576, 2048, 1), 0); del buf60  # reuse
    cpp_fused_relu_18(c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf59, (2048, 512), (512, 1), 0); del buf59  # reuse
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 512), (1, 2048), 0), out=buf62)
    del arg45_1
    buf63 = buf58; del buf58  # reuse
    buf64 = reinterpret_tensor(buf56, (4, 512, 512), (262144, 512, 1), 0); del buf56  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_19(c_void_p(buf57.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg4_1
    buf65 = buf40; del buf40  # reuse
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (2048, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf65)
    del arg46_1
    buf66 = reinterpret_tensor(buf45, (2048, 512), (512, 1), 0); del buf45  # reuse
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (2048, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 512), (1, 512), 0), out=buf66)
    del arg47_1
    buf67 = reinterpret_tensor(buf44, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf44  # reuse
    buf68 = reinterpret_tensor(buf43, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf43  # reuse
    cpp_fused_clone_20(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    buf69 = reinterpret_tensor(buf52, (32, 512, 512), (262144, 512, 1), 0); del buf52  # reuse
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf68, (32, 64, 512), (32768, 512, 1), 0), out=buf69)
    buf70 = buf50; del buf50  # reuse
    buf71 = reinterpret_tensor(buf69, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf69  # reuse
    buf72 = buf48; del buf48  # reuse
    cpp_fused__softmax_21(c_void_p(buf71.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf68, (2048, 512), (512, 1), 0); del buf68  # reuse
    # Source Nodes: [l__mod___model_encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (2048, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf73)
    del arg48_1
    buf74 = buf71; del buf71  # reuse
    buf75 = reinterpret_tensor(buf64, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf64  # reuse
    cpp_fused__softmax_clone_22(c_void_p(buf74.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf73, (32, 512, 64), (32768, 64, 1), 0); del buf73  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf75, (32, 512, 64), (32768, 64, 1), 0), out=buf76)
    buf77 = reinterpret_tensor(buf75, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf75  # reuse
    cpp_fused_clone_23(c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = reinterpret_tensor(buf76, (2048, 512), (512, 1), 0); del buf76  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (2048, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 512), (1, 512), 0), out=buf78)
    del arg49_1
    buf79 = buf63; del buf63  # reuse
    buf80 = reinterpret_tensor(buf77, (4, 512, 512), (262144, 512, 1), 0); del buf77  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_24(c_void_p(buf57.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg5_1
    buf81 = reinterpret_tensor(buf61, (2048, 2048), (2048, 1), 0); del buf61  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (2048, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 2048), (1, 512), 0), out=buf81)
    del arg50_1
    buf82 = reinterpret_tensor(buf81, (4, 512, 2048), (1048576, 2048, 1), 0); del buf81  # reuse
    cpp_fused_relu_25(c_void_p(buf82.data_ptr()))
    buf83 = reinterpret_tensor(buf80, (2048, 512), (512, 1), 0); del buf80  # reuse
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 512), (1, 2048), 0), out=buf83)
    del arg51_1
    buf84 = buf79; del buf79  # reuse
    buf85 = reinterpret_tensor(buf67, (4, 512, 512), (262144, 512, 1), 0); del buf67  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_26(c_void_p(buf57.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg6_1
    buf86 = buf66; del buf66  # reuse
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (2048, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf86)
    del arg52_1
    buf87 = buf65; del buf65  # reuse
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (2048, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf87)
    del arg53_1
    buf88 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf89 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_27(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf74, (32, 512, 512), (262144, 512, 1), 0); del buf74  # reuse
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf89, (32, 64, 512), (32768, 512, 1), 0), out=buf90)
    buf91 = buf72; del buf72  # reuse
    buf92 = reinterpret_tensor(buf90, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf90  # reuse
    buf93 = buf70; del buf70  # reuse
    cpp_fused__softmax_28(c_void_p(buf92.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf89, (2048, 512), (512, 1), 0); del buf89  # reuse
    # Source Nodes: [l__mod___model_encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (2048, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf94)
    del arg54_1
    buf95 = buf92; del buf92  # reuse
    buf96 = reinterpret_tensor(buf85, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf85  # reuse
    cpp_fused__softmax_clone_29(c_void_p(buf95.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = reinterpret_tensor(buf94, (32, 512, 64), (32768, 64, 1), 0); del buf94  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf96, (32, 512, 64), (32768, 64, 1), 0), out=buf97)
    buf98 = reinterpret_tensor(buf96, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf96  # reuse
    cpp_fused_clone_30(c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = reinterpret_tensor(buf97, (2048, 512), (512, 1), 0); del buf97  # reuse
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (2048, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf99)
    del arg55_1
    buf100 = buf57; del buf57  # reuse
    buf101 = buf84; del buf84  # reuse
    buf102 = reinterpret_tensor(buf98, (4, 512, 512), (262144, 512, 1), 0); del buf98  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_31(c_void_p(buf100.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg7_1
    buf103 = reinterpret_tensor(buf82, (2048, 2048), (2048, 1), 0); del buf82  # reuse
    # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (2048, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf103)
    del arg56_1
    buf104 = reinterpret_tensor(buf103, (4, 512, 2048), (1048576, 2048, 1), 0); del buf103  # reuse
    cpp_fused_relu_32(c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf102, (2048, 512), (512, 1), 0); del buf102  # reuse
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 512), (1, 2048), 0), out=buf105)
    del arg57_1
    buf106 = buf101; del buf101  # reuse
    buf107 = reinterpret_tensor(buf99, (4, 512, 512), (262144, 512, 1), 0); del buf99  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_33(c_void_p(buf100.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del arg8_1
    buf108 = buf83; del buf83  # reuse
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (2048, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf108)
    del arg58_1
    buf109 = buf78; del buf78  # reuse
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (2048, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf109)
    del arg59_1
    buf110 = reinterpret_tensor(buf62, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf62  # reuse
    buf111 = reinterpret_tensor(buf88, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf88  # reuse
    cpp_fused_clone_34(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf95, (32, 512, 512), (262144, 512, 1), 0); del buf95  # reuse
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf111, (32, 64, 512), (32768, 512, 1), 0), out=buf112)
    buf113 = buf93; del buf93  # reuse
    buf114 = reinterpret_tensor(buf112, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf112  # reuse
    buf115 = buf91; del buf91  # reuse
    cpp_fused__softmax_35(c_void_p(buf114.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf111, (2048, 512), (512, 1), 0); del buf111  # reuse
    # Source Nodes: [l__mod___model_encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (2048, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf116)
    del arg60_1
    buf117 = buf114; del buf114  # reuse
    buf118 = reinterpret_tensor(buf107, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf107  # reuse
    cpp_fused__softmax_clone_36(c_void_p(buf117.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()))
    buf119 = reinterpret_tensor(buf116, (32, 512, 64), (32768, 64, 1), 0); del buf116  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf117, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf118, (32, 512, 64), (32768, 64, 1), 0), out=buf119)
    buf120 = reinterpret_tensor(buf118, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf118  # reuse
    cpp_fused_clone_37(c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf119, (2048, 512), (512, 1), 0); del buf119  # reuse
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (2048, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf121)
    del arg61_1
    buf122 = buf106; del buf106  # reuse
    buf123 = reinterpret_tensor(buf120, (4, 512, 512), (262144, 512, 1), 0); del buf120  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_38(c_void_p(buf100.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg9_1
    buf124 = reinterpret_tensor(buf104, (2048, 2048), (2048, 1), 0); del buf104  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (2048, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 2048), (1, 512), 0), out=buf124)
    del arg62_1
    buf125 = reinterpret_tensor(buf124, (4, 512, 2048), (1048576, 2048, 1), 0); del buf124  # reuse
    cpp_fused_relu_39(c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf123, (2048, 512), (512, 1), 0); del buf123  # reuse
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 512), (1, 2048), 0), out=buf126)
    del arg63_1
    buf127 = buf122; del buf122  # reuse
    buf128 = reinterpret_tensor(buf110, (4, 512, 512), (262144, 512, 1), 0); del buf110  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_40(c_void_p(buf100.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg10_1
    buf129 = buf109; del buf109  # reuse
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (2048, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf129)
    del arg64_1
    buf130 = buf108; del buf108  # reuse
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (2048, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 512), (1, 512), 0), out=buf130)
    del arg65_1
    buf131 = reinterpret_tensor(buf87, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf87  # reuse
    buf132 = reinterpret_tensor(buf86, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf86  # reuse
    cpp_fused_clone_41(c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf117, (32, 512, 512), (262144, 512, 1), 0); del buf117  # reuse
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf132, (32, 64, 512), (32768, 512, 1), 0), out=buf133)
    buf134 = buf115; del buf115  # reuse
    buf135 = reinterpret_tensor(buf133, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf133  # reuse
    buf136 = buf113; del buf113  # reuse
    cpp_fused__softmax_42(c_void_p(buf135.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg36_1
    buf137 = reinterpret_tensor(buf132, (2048, 512), (512, 1), 0); del buf132  # reuse
    # Source Nodes: [l__mod___model_encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (2048, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf137)
    del arg66_1
    buf138 = buf135; del buf135  # reuse
    buf139 = reinterpret_tensor(buf128, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf128  # reuse
    cpp_fused__softmax_clone_43(c_void_p(buf138.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf137, (32, 512, 64), (32768, 64, 1), 0); del buf137  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf139, (32, 512, 64), (32768, 64, 1), 0), out=buf140)
    buf141 = reinterpret_tensor(buf139, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf139  # reuse
    cpp_fused_clone_44(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf140, (2048, 512), (512, 1), 0); del buf140  # reuse
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf141, (2048, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 512), (1, 512), 0), out=buf142)
    del arg67_1
    buf143 = buf100; del buf100  # reuse
    buf144 = buf127; del buf127  # reuse
    buf145 = reinterpret_tensor(buf141, (4, 512, 512), (262144, 512, 1), 0); del buf141  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_45(c_void_p(buf143.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del arg11_1
    buf146 = reinterpret_tensor(buf125, (2048, 2048), (2048, 1), 0); del buf125  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (2048, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 2048), (1, 512), 0), out=buf146)
    del arg68_1
    buf147 = reinterpret_tensor(buf146, (4, 512, 2048), (1048576, 2048, 1), 0); del buf146  # reuse
    cpp_fused_relu_46(c_void_p(buf147.data_ptr()))
    buf148 = reinterpret_tensor(buf145, (2048, 512), (512, 1), 0); del buf145  # reuse
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 512), (1, 2048), 0), out=buf148)
    del arg69_1
    buf149 = buf144; del buf144  # reuse
    buf150 = buf143; del buf143  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_47(c_void_p(buf150.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf149.data_ptr()))
    del arg12_1
    buf151 = buf148; del buf148  # reuse
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 512), (1, 512), 0), out=buf151)
    del arg76_1
    buf152 = reinterpret_tensor(buf142, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf142  # reuse
    buf153 = reinterpret_tensor(buf126, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf126  # reuse
    cpp_fused_clone_48(c_void_p(buf19.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    buf154 = reinterpret_tensor(buf138, (32, 512, 512), (262144, 512, 1), 0); del buf138  # reuse
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf153, (32, 64, 512), (32768, 512, 1), 0), out=buf154)
    buf155 = buf136; del buf136  # reuse
    buf156 = reinterpret_tensor(buf154, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf154  # reuse
    buf157 = buf134; del buf134  # reuse
    cpp_fused__softmax_49(c_void_p(buf156.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = reinterpret_tensor(buf153, (2048, 512), (512, 1), 0); del buf153  # reuse
    # Source Nodes: [l__mod___model_decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 512), (1, 512), 0), out=buf158)
    del arg77_1
    buf159 = buf156; del buf156  # reuse
    buf160 = buf152; del buf152  # reuse
    cpp_fused__softmax_clone_50(c_void_p(buf159.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf19, (32, 512, 64), (32768, 64, 1), 0); del buf19  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf159, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf160, (32, 512, 64), (32768, 64, 1), 0), out=buf161)
    buf162 = reinterpret_tensor(buf160, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf160  # reuse
    cpp_fused_clone_51(c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    buf163 = reinterpret_tensor(buf161, (2048, 512), (512, 1), 0); del buf161  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf162, (2048, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 512), (1, 512), 0), out=buf163)
    del arg78_1
    buf164 = buf149; del buf149  # reuse
    buf165 = reinterpret_tensor(buf162, (4, 512, 512), (262144, 512, 1), 0); del buf162  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_52(c_void_p(arg133_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg15_1
    buf166 = reinterpret_tensor(buf147, (2048, 2048), (2048, 1), 0); del buf147  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (2048, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf166)
    del arg79_1
    buf167 = reinterpret_tensor(buf166, (4, 512, 2048), (1048576, 2048, 1), 0); del buf166  # reuse
    cpp_fused_relu_53(c_void_p(buf167.data_ptr()))
    buf168 = reinterpret_tensor(buf165, (2048, 512), (512, 1), 0); del buf165  # reuse
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg80_1, (2048, 512), (1, 2048), 0), out=buf168)
    del arg80_1
    buf169 = reinterpret_tensor(buf16, (4, 512, 512), (262144, 512, 1), 0); del buf16  # reuse
    buf170 = buf164; del buf164  # reuse
    buf171 = reinterpret_tensor(buf121, (4, 512, 512), (262144, 512, 1), 0); del buf121  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_54(c_void_p(buf169.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del arg133_1
    del arg16_1
    del arg32_1
    buf172 = buf168; del buf168  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (2048, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 512), (1, 512), 0), out=buf172)
    del arg81_1
    buf173 = buf163; del buf163  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (2048, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 512), (1, 512), 0), out=buf173)
    del arg82_1
    buf174 = reinterpret_tensor(buf105, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf105  # reuse
    buf175 = reinterpret_tensor(buf131, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf131  # reuse
    cpp_fused_clone_55(c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf159, (32, 512, 512), (262144, 512, 1), 0); del buf159  # reuse
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf174, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf175, (32, 64, 512), (32768, 512, 1), 0), out=buf176)
    buf177 = buf157; del buf157  # reuse
    buf178 = reinterpret_tensor(buf176, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf176  # reuse
    buf179 = buf178; del buf178  # reuse
    buf180 = buf155; del buf155  # reuse
    cpp_fused__softmax_56(c_void_p(buf179.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf175, (2048, 512), (512, 1), 0); del buf175  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (2048, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 512), (1, 512), 0), out=buf181)
    del arg83_1
    buf182 = buf179; del buf179  # reuse
    buf183 = reinterpret_tensor(buf171, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf171  # reuse
    cpp_fused__softmax_clone_57(c_void_p(buf182.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = reinterpret_tensor(buf174, (32, 512, 64), (32768, 64, 1), 0); del buf174  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf182, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf183, (32, 512, 64), (32768, 64, 1), 0), out=buf184)
    buf185 = reinterpret_tensor(buf183, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf183  # reuse
    cpp_fused_clone_58(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = reinterpret_tensor(buf184, (2048, 512), (512, 1), 0); del buf184  # reuse
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf185, (2048, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf186)
    del arg84_1
    buf187 = buf170; del buf170  # reuse
    buf188 = reinterpret_tensor(buf185, (4, 512, 512), (262144, 512, 1), 0); del buf185  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_59(c_void_p(buf169.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg17_1
    buf189 = buf172; del buf172  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (2048, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf189)
    del arg85_1
    buf190 = reinterpret_tensor(buf188, (2048, 512), (512, 1), 0); del buf188  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf190)
    del arg86_1
    buf191 = reinterpret_tensor(buf130, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf130  # reuse
    buf192 = reinterpret_tensor(buf129, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf129  # reuse
    cpp_fused_clone_60(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = reinterpret_tensor(buf182, (32, 512, 512), (262144, 512, 1), 0); del buf182  # reuse
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf191, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf192, (32, 64, 512), (32768, 512, 1), 0), out=buf193)
    buf194 = buf180; del buf180  # reuse
    buf195 = reinterpret_tensor(buf193, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf193  # reuse
    buf196 = buf177; del buf177  # reuse
    cpp_fused__softmax_61(c_void_p(buf195.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()))
    buf197 = reinterpret_tensor(buf192, (2048, 512), (512, 1), 0); del buf192  # reuse
    # Source Nodes: [l__mod___model_decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf197)
    del arg87_1
    buf198 = buf195; del buf195  # reuse
    buf199 = buf191; del buf191  # reuse
    cpp_fused__softmax_clone_62(c_void_p(buf198.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf189, (32, 512, 64), (32768, 64, 1), 0); del buf189  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf198, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf199, (32, 512, 64), (32768, 64, 1), 0), out=buf200)
    buf201 = reinterpret_tensor(buf199, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf199  # reuse
    cpp_fused_clone_63(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = reinterpret_tensor(buf200, (2048, 512), (512, 1), 0); del buf200  # reuse
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (2048, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf202)
    del arg88_1
    buf203 = buf187; del buf187  # reuse
    buf204 = reinterpret_tensor(buf201, (4, 512, 512), (262144, 512, 1), 0); del buf201  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_64(c_void_p(buf169.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del arg18_1
    buf205 = reinterpret_tensor(buf167, (2048, 2048), (2048, 1), 0); del buf167  # reuse
    # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (2048, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf205)
    del arg89_1
    buf206 = reinterpret_tensor(buf205, (4, 512, 2048), (1048576, 2048, 1), 0); del buf205  # reuse
    cpp_fused_relu_65(c_void_p(buf206.data_ptr()))
    buf207 = reinterpret_tensor(buf204, (2048, 512), (512, 1), 0); del buf204  # reuse
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf207)
    del arg90_1
    buf208 = buf203; del buf203  # reuse
    buf209 = empty((4, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_66(c_void_p(buf169.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg19_1
    buf210 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (2048, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf210)
    del arg91_1
    buf211 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (2048, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 512), (1, 512), 0), out=buf211)
    del arg92_1
    buf212 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf213 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_67(c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf198, (32, 512, 512), (262144, 512, 1), 0); del buf198  # reuse
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf212, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf213, (32, 64, 512), (32768, 512, 1), 0), out=buf214)
    buf215 = buf196; del buf196  # reuse
    buf216 = reinterpret_tensor(buf214, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf214  # reuse
    buf217 = buf216; del buf216  # reuse
    buf218 = buf194; del buf194  # reuse
    cpp_fused__softmax_68(c_void_p(buf217.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf213, (2048, 512), (512, 1), 0); del buf213  # reuse
    # Source Nodes: [l__mod___model_decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (2048, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf219)
    del arg93_1
    buf220 = buf217; del buf217  # reuse
    buf221 = reinterpret_tensor(buf209, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf209  # reuse
    cpp_fused__softmax_clone_69(c_void_p(buf220.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf212, (32, 512, 64), (32768, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf221, (32, 512, 64), (32768, 64, 1), 0), out=buf222)
    buf223 = reinterpret_tensor(buf221, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf221  # reuse
    cpp_fused_clone_70(c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf222, (2048, 512), (512, 1), 0); del buf222  # reuse
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (2048, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 512), (1, 512), 0), out=buf224)
    del arg94_1
    buf225 = buf169; del buf169  # reuse
    buf226 = buf208; del buf208  # reuse
    buf227 = reinterpret_tensor(buf223, (4, 512, 512), (262144, 512, 1), 0); del buf223  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_71(c_void_p(buf225.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg20_1
    buf228 = buf224; del buf224  # reuse
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (2048, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 512), (1, 512), 0), out=buf228)
    del arg95_1
    buf229 = reinterpret_tensor(buf227, (2048, 512), (512, 1), 0); del buf227  # reuse
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf229)
    del arg96_1
    buf230 = reinterpret_tensor(buf207, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf207  # reuse
    buf231 = reinterpret_tensor(buf202, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf202  # reuse
    cpp_fused_clone_72(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    buf232 = reinterpret_tensor(buf220, (32, 512, 512), (262144, 512, 1), 0); del buf220  # reuse
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf231, (32, 64, 512), (32768, 512, 1), 0), out=buf232)
    buf233 = buf218; del buf218  # reuse
    buf234 = reinterpret_tensor(buf232, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf232  # reuse
    buf235 = buf215; del buf215  # reuse
    cpp_fused__softmax_73(c_void_p(buf234.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf231, (2048, 512), (512, 1), 0); del buf231  # reuse
    # Source Nodes: [l__mod___model_decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 512), (1, 512), 0), out=buf236)
    del arg97_1
    buf237 = buf234; del buf234  # reuse
    buf238 = buf230; del buf230  # reuse
    cpp_fused__softmax_clone_74(c_void_p(buf237.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = reinterpret_tensor(buf228, (32, 512, 64), (32768, 64, 1), 0); del buf228  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf237, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf238, (32, 512, 64), (32768, 64, 1), 0), out=buf239)
    buf240 = reinterpret_tensor(buf238, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf238  # reuse
    cpp_fused_clone_75(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf239, (2048, 512), (512, 1), 0); del buf239  # reuse
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (2048, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf241)
    del arg98_1
    buf242 = buf226; del buf226  # reuse
    buf243 = reinterpret_tensor(buf240, (4, 512, 512), (262144, 512, 1), 0); del buf240  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_76(c_void_p(buf225.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    del arg21_1
    buf244 = reinterpret_tensor(buf206, (2048, 2048), (2048, 1), 0); del buf206  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (2048, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 2048), (1, 512), 0), out=buf244)
    del arg99_1
    buf245 = reinterpret_tensor(buf244, (4, 512, 2048), (1048576, 2048, 1), 0); del buf244  # reuse
    cpp_fused_relu_77(c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf243, (2048, 512), (512, 1), 0); del buf243  # reuse
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 512), (1, 2048), 0), out=buf246)
    del arg100_1
    buf247 = buf242; del buf242  # reuse
    buf248 = reinterpret_tensor(buf186, (4, 512, 512), (262144, 512, 1), 0); del buf186  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_78(c_void_p(buf225.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg22_1
    buf249 = buf210; del buf210  # reuse
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (2048, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf249)
    del arg101_1
    buf250 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (2048, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf250)
    del arg102_1
    buf251 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf252 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_79(c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    buf253 = reinterpret_tensor(buf237, (32, 512, 512), (262144, 512, 1), 0); del buf237  # reuse
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf251, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf252, (32, 64, 512), (32768, 512, 1), 0), out=buf253)
    buf254 = buf235; del buf235  # reuse
    buf255 = reinterpret_tensor(buf253, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf253  # reuse
    buf256 = buf255; del buf255  # reuse
    buf257 = buf233; del buf233  # reuse
    cpp_fused__softmax_80(c_void_p(buf256.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()))
    buf258 = reinterpret_tensor(buf252, (2048, 512), (512, 1), 0); del buf252  # reuse
    # Source Nodes: [l__mod___model_decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (2048, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), out=buf258)
    del arg103_1
    buf259 = buf256; del buf256  # reuse
    buf260 = reinterpret_tensor(buf248, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf248  # reuse
    cpp_fused__softmax_clone_81(c_void_p(buf259.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = reinterpret_tensor(buf251, (32, 512, 64), (32768, 64, 1), 0); del buf251  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf260, (32, 512, 64), (32768, 64, 1), 0), out=buf261)
    buf262 = reinterpret_tensor(buf260, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf260  # reuse
    cpp_fused_clone_82(c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = reinterpret_tensor(buf261, (2048, 512), (512, 1), 0); del buf261  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (2048, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf263)
    del arg104_1
    buf264 = buf247; del buf247  # reuse
    buf265 = reinterpret_tensor(buf262, (4, 512, 512), (262144, 512, 1), 0); del buf262  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_83(c_void_p(buf225.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del arg23_1
    buf266 = buf249; del buf249  # reuse
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf265, (2048, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), out=buf266)
    del arg105_1
    buf267 = reinterpret_tensor(buf265, (2048, 512), (512, 1), 0); del buf265  # reuse
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf267)
    del arg106_1
    buf268 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf269 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_84(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf259, (32, 512, 512), (262144, 512, 1), 0); del buf259  # reuse
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf269, (32, 64, 512), (32768, 512, 1), 0), out=buf270)
    buf271 = buf257; del buf257  # reuse
    buf272 = reinterpret_tensor(buf270, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf270  # reuse
    buf273 = buf254; del buf254  # reuse
    cpp_fused__softmax_85(c_void_p(buf272.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf269, (2048, 512), (512, 1), 0); del buf269  # reuse
    # Source Nodes: [l__mod___model_decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf274)
    del arg107_1
    buf275 = buf272; del buf272  # reuse
    buf276 = buf268; del buf268  # reuse
    cpp_fused__softmax_clone_86(c_void_p(buf275.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = reinterpret_tensor(buf266, (32, 512, 64), (32768, 64, 1), 0); del buf266  # reuse
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf275, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf276, (32, 512, 64), (32768, 64, 1), 0), out=buf277)
    buf278 = reinterpret_tensor(buf276, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf276  # reuse
    cpp_fused_clone_87(c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    buf279 = reinterpret_tensor(buf277, (2048, 512), (512, 1), 0); del buf277  # reuse
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf278, (2048, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf279)
    del arg108_1
    buf280 = buf225; del buf225  # reuse
    buf281 = buf264; del buf264  # reuse
    buf282 = reinterpret_tensor(buf278, (4, 512, 512), (262144, 512, 1), 0); del buf278  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_88(c_void_p(buf280.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del arg24_1
    buf283 = reinterpret_tensor(buf245, (2048, 2048), (2048, 1), 0); del buf245  # reuse
    # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (2048, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 2048), (1, 512), 0), out=buf283)
    del arg109_1
    buf284 = reinterpret_tensor(buf283, (4, 512, 2048), (1048576, 2048, 1), 0); del buf283  # reuse
    cpp_fused_relu_89(c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf282, (2048, 512), (512, 1), 0); del buf282  # reuse
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf284, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 512), (1, 2048), 0), out=buf285)
    del arg110_1
    buf286 = buf281; del buf281  # reuse
    buf287 = reinterpret_tensor(buf279, (4, 512, 512), (262144, 512, 1), 0); del buf279  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_90(c_void_p(buf280.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del arg25_1
    buf288 = buf263; del buf263  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (2048, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 512), (1, 512), 0), out=buf288)
    del arg111_1
    buf289 = buf246; del buf246  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (2048, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf289)
    del arg112_1
    buf290 = reinterpret_tensor(buf241, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf241  # reuse
    buf291 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_91(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = reinterpret_tensor(buf275, (32, 512, 512), (262144, 512, 1), 0); del buf275  # reuse
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf290, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf291, (32, 64, 512), (32768, 512, 1), 0), out=buf292)
    buf293 = buf273; del buf273  # reuse
    buf294 = reinterpret_tensor(buf292, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf292  # reuse
    buf295 = buf294; del buf294  # reuse
    buf296 = buf271; del buf271  # reuse
    cpp_fused__softmax_92(c_void_p(buf295.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf296.data_ptr()))
    buf297 = reinterpret_tensor(buf291, (2048, 512), (512, 1), 0); del buf291  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (2048, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 512), (1, 512), 0), out=buf297)
    del arg113_1
    buf298 = buf295; del buf295  # reuse
    buf299 = reinterpret_tensor(buf287, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf287  # reuse
    cpp_fused__softmax_clone_93(c_void_p(buf298.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf290, (32, 512, 64), (32768, 64, 1), 0); del buf290  # reuse
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf298, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf299, (32, 512, 64), (32768, 64, 1), 0), out=buf300)
    buf301 = reinterpret_tensor(buf299, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf299  # reuse
    cpp_fused_clone_94(c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    buf302 = reinterpret_tensor(buf300, (2048, 512), (512, 1), 0); del buf300  # reuse
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (2048, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf302)
    del arg114_1
    buf303 = buf286; del buf286  # reuse
    buf304 = reinterpret_tensor(buf301, (4, 512, 512), (262144, 512, 1), 0); del buf301  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_95(c_void_p(buf280.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del arg26_1
    buf305 = buf288; del buf288  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (2048, 512), (512, 1), 0), reinterpret_tensor(arg115_1, (512, 512), (1, 512), 0), out=buf305)
    del arg115_1
    buf306 = reinterpret_tensor(buf304, (2048, 512), (512, 1), 0); del buf304  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 512), (1, 512), 0), out=buf306)
    del arg116_1
    buf307 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf308 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_96(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = reinterpret_tensor(buf298, (32, 512, 512), (262144, 512, 1), 0); del buf298  # reuse
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf307, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf308, (32, 64, 512), (32768, 512, 1), 0), out=buf309)
    buf310 = buf296; del buf296  # reuse
    buf311 = reinterpret_tensor(buf309, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf309  # reuse
    buf312 = buf293; del buf293  # reuse
    cpp_fused__softmax_97(c_void_p(buf311.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()))
    buf313 = reinterpret_tensor(buf308, (2048, 512), (512, 1), 0); del buf308  # reuse
    # Source Nodes: [l__mod___model_decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), out=buf313)
    del arg117_1
    buf314 = buf311; del buf311  # reuse
    buf315 = buf307; del buf307  # reuse
    cpp_fused__softmax_clone_98(c_void_p(buf314.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = reinterpret_tensor(buf305, (32, 512, 64), (32768, 64, 1), 0); del buf305  # reuse
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf314, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf315, (32, 512, 64), (32768, 64, 1), 0), out=buf316)
    buf317 = reinterpret_tensor(buf315, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf315  # reuse
    cpp_fused_clone_99(c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = reinterpret_tensor(buf316, (2048, 512), (512, 1), 0); del buf316  # reuse
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (2048, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 512), (1, 512), 0), out=buf318)
    del arg118_1
    buf319 = buf303; del buf303  # reuse
    buf320 = reinterpret_tensor(buf317, (4, 512, 512), (262144, 512, 1), 0); del buf317  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_100(c_void_p(buf280.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del arg27_1
    buf321 = reinterpret_tensor(buf284, (2048, 2048), (2048, 1), 0); del buf284  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf320, (2048, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 2048), (1, 512), 0), out=buf321)
    del arg119_1
    buf322 = reinterpret_tensor(buf321, (4, 512, 2048), (1048576, 2048, 1), 0); del buf321  # reuse
    cpp_fused_relu_101(c_void_p(buf322.data_ptr()))
    buf323 = reinterpret_tensor(buf320, (2048, 512), (512, 1), 0); del buf320  # reuse
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 512), (1, 2048), 0), out=buf323)
    del arg120_1
    buf324 = buf280; del buf280  # reuse
    buf325 = buf319; del buf319  # reuse
    buf326 = empty((4, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_102(c_void_p(buf324.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del arg28_1
    buf327 = buf323; del buf323  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (2048, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf327)
    del arg121_1
    buf328 = buf318; del buf318  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (2048, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf328)
    del arg122_1
    buf329 = reinterpret_tensor(buf302, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf302  # reuse
    buf330 = reinterpret_tensor(buf285, (4, 8, 64, 512), (262144, 32768, 512, 1), 0); del buf285  # reuse
    cpp_fused_clone_103(c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = reinterpret_tensor(buf314, (32, 512, 512), (262144, 512, 1), 0); del buf314  # reuse
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf330, (32, 64, 512), (32768, 512, 1), 0), out=buf331)
    buf332 = buf312; del buf312  # reuse
    buf333 = reinterpret_tensor(buf331, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf331  # reuse
    buf334 = buf333; del buf333  # reuse
    buf335 = buf310; del buf310  # reuse
    cpp_fused__softmax_104(c_void_p(buf334.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf335.data_ptr()))
    del arg73_1
    buf336 = reinterpret_tensor(buf330, (2048, 512), (512, 1), 0); del buf330  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (2048, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), out=buf336)
    del arg123_1
    buf337 = buf334; del buf334  # reuse
    buf338 = reinterpret_tensor(buf326, (4, 8, 512, 64), (262144, 32768, 64, 1), 0); del buf326  # reuse
    cpp_fused__softmax_clone_105(c_void_p(buf337.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()))
    buf339 = reinterpret_tensor(buf329, (32, 512, 64), (32768, 64, 1), 0); del buf329  # reuse
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf337, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf338, (32, 512, 64), (32768, 64, 1), 0), out=buf339)
    buf340 = reinterpret_tensor(buf338, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf338  # reuse
    cpp_fused_clone_106(c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = reinterpret_tensor(buf339, (2048, 512), (512, 1), 0); del buf339  # reuse
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf340, (2048, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf341)
    del arg124_1
    buf342 = buf325; del buf325  # reuse
    buf343 = reinterpret_tensor(buf340, (4, 512, 512), (262144, 512, 1), 0); del buf340  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_107(c_void_p(buf324.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    del arg29_1
    buf344 = buf327; del buf327  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (2048, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 512), (1, 512), 0), out=buf344)
    del arg125_1
    buf345 = reinterpret_tensor(buf343, (2048, 512), (512, 1), 0); del buf343  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf345)
    del arg126_1
    buf346 = empty((4, 8, 512, 64), device='cpu', dtype=torch.float32)
    buf347 = empty((4, 8, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_108(c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = reinterpret_tensor(buf337, (32, 512, 512), (262144, 512, 1), 0); del buf337  # reuse
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf346, (32, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf347, (32, 64, 512), (32768, 512, 1), 0), out=buf348)
    buf349 = buf335; del buf335  # reuse
    buf350 = reinterpret_tensor(buf348, (4, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf348  # reuse
    buf351 = buf332; del buf332  # reuse
    cpp_fused__softmax_109(c_void_p(buf350.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del buf349
    buf352 = reinterpret_tensor(buf347, (2048, 512), (512, 1), 0); del buf347  # reuse
    # Source Nodes: [l__mod___model_decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (2048, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 512), (1, 512), 0), out=buf352)
    del arg127_1
    buf353 = buf350; del buf350  # reuse
    buf354 = buf346; del buf346  # reuse
    cpp_fused__softmax_clone_110(c_void_p(buf353.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()))
    del buf351
    buf355 = reinterpret_tensor(buf344, (32, 512, 64), (32768, 64, 1), 0); del buf344  # reuse
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (32, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf354, (32, 512, 64), (32768, 64, 1), 0), out=buf355)
    del buf353
    buf356 = reinterpret_tensor(buf354, (4, 512, 8, 64), (262144, 512, 64, 1), 0); del buf354  # reuse
    cpp_fused_clone_111(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = reinterpret_tensor(buf355, (2048, 512), (512, 1), 0); del buf355  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (2048, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 512), (1, 512), 0), out=buf357)
    del arg128_1
    buf358 = buf342; del buf342  # reuse
    buf359 = reinterpret_tensor(buf356, (4, 512, 512), (262144, 512, 1), 0); del buf356  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_112(c_void_p(buf324.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del arg30_1
    buf360 = reinterpret_tensor(buf322, (2048, 2048), (2048, 1), 0); del buf322  # reuse
    # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (2048, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 2048), (1, 512), 0), out=buf360)
    del arg129_1
    buf361 = reinterpret_tensor(buf360, (4, 512, 2048), (1048576, 2048, 1), 0); del buf360  # reuse
    cpp_fused_relu_113(c_void_p(buf361.data_ptr()))
    buf362 = reinterpret_tensor(buf359, (2048, 512), (512, 1), 0); del buf359  # reuse
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf362)
    del arg130_1
    del buf361
    buf363 = buf358; del buf358  # reuse
    buf364 = buf324; del buf324  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_114(c_void_p(buf364.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf363.data_ptr()))
    del arg31_1
    del buf341
    del buf357
    del buf362
    del buf363
    buf365 = empty((2048, 32128), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (2048, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 32128), (1, 512), 0), out=buf365)
    del arg131_1
    return (reinterpret_tensor(buf365, (4, 512, 32128), (16449536, 32128, 1), 0), reinterpret_tensor(buf3, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf11, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf151, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf158, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf173, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf181, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf190, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf197, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf211, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf219, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf229, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf236, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf250, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf258, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf267, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf274, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf289, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf297, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf306, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf313, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf328, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf336, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf345, (4, 8, 512, 64), (262144, 64, 512, 1), 0), reinterpret_tensor(buf352, (4, 8, 512, 64), (262144, 64, 512, 1), 0), buf150, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((32, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((32, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg133_1 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
