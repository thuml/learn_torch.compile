
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_1 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_5 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_11 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__softmax_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_23 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_29 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__softmax_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_41 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_47 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_51 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_53 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_57 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_59 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__softmax_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0,
                       float* out_ptr1)
{
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_67 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_69 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_71 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_73 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_77 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_81 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_83 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_87 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_89 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_91 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_93 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (65536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_relu_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__log_softmax_nll_loss_forward_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 32128);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 32128L), "index out of bounds: 0 <= tmp7 < 32128L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (32128L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp12 = std::log(tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 - tmp12);
                        auto tmp14 = decltype(tmp13)(-tmp13);
                        auto tmp15 = static_cast<float>(0.0);
                        auto tmp16 = tmp2 ? tmp14 : tmp15;
                        auto tmp17 = c10::convert<long>(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp17;
                    }
                    out_ptr2[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr3[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr2[static_cast<long>(0L)];
                auto tmp1 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1 = args
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
    assert_size_stride(arg132_1, (1, 1024), (1024, 1))
    assert_size_stride(arg133_1, (1, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1, 1024), (1024, 1))
    buf0 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(c_void_p(arg134_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg13_1
    buf2 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf2)
    del arg70_1
    buf3 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), out=buf3)
    del arg71_1
    buf4 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf3, (8, 64, 1024), (64, 1, 512), 0), out=buf4)
    buf5 = empty_strided((1, 8, 1024, 1), (8192, 1024, 1, 8192), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf4, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf4  # reuse
    buf7 = buf6; del buf6  # reuse
    buf8 = empty_strided((1, 8, 1024, 1), (8192, 1024, 1, 8192), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_1(c_void_p(buf7.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf8.data_ptr()))
    buf9 = buf2; del buf2  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf9)
    del arg72_1
    buf10 = buf7; del buf7  # reuse
    cpp_fused__softmax_2(c_void_p(buf10.data_ptr()), c_void_p(buf8.data_ptr()))
    buf11 = reinterpret_tensor(buf1, (8, 1024, 64), (65536, 64, 1), 0); del buf1  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf10, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf9, (8, 1024, 64), (64, 512, 1), 0), out=buf11)
    buf12 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_3(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = reinterpret_tensor(buf11, (1024, 512), (512, 1), 0); del buf11  # reuse
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (1024, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf13)
    del arg74_1
    buf14 = buf0; del buf0  # reuse
    buf17 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf12, (1, 1024, 512), (524288, 512, 1), 0); del buf12  # reuse
    buf18 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_4(c_void_p(arg134_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg0_1
    del arg14_1
    buf16 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (1024, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf16)
    del arg75_1
    buf19 = reinterpret_tensor(buf15, (1024, 512), (512, 1), 0); del buf15  # reuse
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 512), (512, 1), 0), reinterpret_tensor(arg33_1, (512, 512), (1, 512), 0), out=buf19)
    del arg33_1
    buf20 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 512), (1, 512), 0), out=buf20)
    del arg34_1
    buf21 = reinterpret_tensor(buf10, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf10  # reuse
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf19, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf20, (8, 64, 1024), (64, 1, 512), 0), out=buf21)
    buf22 = buf8; del buf8  # reuse
    buf23 = reinterpret_tensor(buf21, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf21  # reuse
    buf24 = buf5; del buf5  # reuse
    cpp_fused__softmax_5(c_void_p(buf23.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = buf20; del buf20  # reuse
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 512), (1, 512), 0), out=buf25)
    del arg35_1
    buf26 = buf23; del buf23  # reuse
    cpp_fused__softmax_6(c_void_p(buf26.data_ptr()), c_void_p(buf24.data_ptr()))
    buf27 = reinterpret_tensor(buf18, (8, 1024, 64), (65536, 64, 1), 0); del buf18  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf25, (8, 1024, 64), (64, 512, 1), 0), out=buf27)
    buf28 = reinterpret_tensor(buf25, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf25  # reuse
    cpp_fused_clone_7(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = reinterpret_tensor(buf27, (1024, 512), (512, 1), 0); del buf27  # reuse
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (1024, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf29)
    del arg37_1
    buf30 = buf17; del buf17  # reuse
    buf31 = reinterpret_tensor(buf28, (1, 1024, 512), (524288, 512, 1), 0); del buf28  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_8(c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg1_1
    buf32 = empty((1024, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (1024, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 2048), (1, 512), 0), out=buf32)
    del arg38_1
    buf33 = reinterpret_tensor(buf32, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf32  # reuse
    cpp_fused_relu_9(c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf31, (1024, 512), (512, 1), 0); del buf31  # reuse
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf33, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 512), (1, 2048), 0), out=buf34)
    del arg39_1
    buf35 = buf30; del buf30  # reuse
    buf36 = reinterpret_tensor(buf19, (1, 1024, 512), (524288, 512, 1), 0); del buf19  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_10(c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg2_1
    buf37 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1024, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf37)
    del arg40_1
    buf38 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1024, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), out=buf38)
    del arg41_1
    buf39 = reinterpret_tensor(buf26, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf26  # reuse
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf37, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf38, (8, 64, 1024), (64, 1, 512), 0), out=buf39)
    buf40 = buf24; del buf24  # reuse
    buf41 = reinterpret_tensor(buf39, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf39  # reuse
    buf42 = buf22; del buf22  # reuse
    cpp_fused__softmax_11(c_void_p(buf41.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = buf38; del buf38  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1024, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 512), (1, 512), 0), out=buf43)
    del arg42_1
    buf44 = buf41; del buf41  # reuse
    cpp_fused__softmax_12(c_void_p(buf44.data_ptr()), c_void_p(buf42.data_ptr()))
    buf45 = reinterpret_tensor(buf36, (8, 1024, 64), (65536, 64, 1), 0); del buf36  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf43, (8, 1024, 64), (64, 512, 1), 0), out=buf45)
    buf46 = reinterpret_tensor(buf43, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf43  # reuse
    cpp_fused_clone_13(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf45, (1024, 512), (512, 1), 0); del buf45  # reuse
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (1024, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf47)
    del arg43_1
    buf48 = reinterpret_tensor(buf29, (1, 1024, 512), (524288, 512, 1), 0); del buf29  # reuse
    buf49 = buf35; del buf35  # reuse
    buf50 = reinterpret_tensor(buf46, (1, 1024, 512), (524288, 512, 1), 0); del buf46  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_14(c_void_p(buf48.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg132_1
    del arg3_1
    buf51 = reinterpret_tensor(buf33, (1024, 2048), (2048, 1), 0); del buf33  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf50, (1024, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 2048), (1, 512), 0), out=buf51)
    del arg44_1
    buf52 = reinterpret_tensor(buf51, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf51  # reuse
    cpp_fused_relu_15(c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf50, (1024, 512), (512, 1), 0); del buf50  # reuse
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 512), (1, 2048), 0), out=buf53)
    del arg45_1
    buf54 = buf49; del buf49  # reuse
    buf55 = reinterpret_tensor(buf47, (1, 1024, 512), (524288, 512, 1), 0); del buf47  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_16(c_void_p(buf48.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg4_1
    buf56 = buf34; del buf34  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1024, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf56)
    del arg46_1
    buf57 = buf37; del buf37  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1024, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 512), (1, 512), 0), out=buf57)
    del arg47_1
    buf58 = reinterpret_tensor(buf44, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf44  # reuse
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf57, (8, 64, 1024), (64, 1, 512), 0), out=buf58)
    buf59 = buf42; del buf42  # reuse
    buf60 = reinterpret_tensor(buf58, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf58  # reuse
    buf61 = buf40; del buf40  # reuse
    cpp_fused__softmax_17(c_void_p(buf60.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = buf57; del buf57  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1024, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf62)
    del arg48_1
    buf63 = buf60; del buf60  # reuse
    cpp_fused__softmax_18(c_void_p(buf63.data_ptr()), c_void_p(buf61.data_ptr()))
    buf64 = reinterpret_tensor(buf55, (8, 1024, 64), (65536, 64, 1), 0); del buf55  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf62, (8, 1024, 64), (64, 512, 1), 0), out=buf64)
    buf65 = reinterpret_tensor(buf62, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf62  # reuse
    cpp_fused_clone_19(c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    buf66 = reinterpret_tensor(buf64, (1024, 512), (512, 1), 0); del buf64  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (1024, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 512), (1, 512), 0), out=buf66)
    del arg49_1
    buf67 = buf54; del buf54  # reuse
    buf68 = reinterpret_tensor(buf65, (1, 1024, 512), (524288, 512, 1), 0); del buf65  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_20(c_void_p(buf48.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg5_1
    buf69 = reinterpret_tensor(buf52, (1024, 2048), (2048, 1), 0); del buf52  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (1024, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 2048), (1, 512), 0), out=buf69)
    del arg50_1
    buf70 = reinterpret_tensor(buf69, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf69  # reuse
    cpp_fused_relu_21(c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf68, (1024, 512), (512, 1), 0); del buf68  # reuse
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 512), (1, 2048), 0), out=buf71)
    del arg51_1
    buf72 = buf67; del buf67  # reuse
    buf73 = reinterpret_tensor(buf56, (1, 1024, 512), (524288, 512, 1), 0); del buf56  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_22(c_void_p(buf48.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg6_1
    buf74 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (1024, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf74)
    del arg52_1
    buf75 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (1024, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf75)
    del arg53_1
    buf76 = reinterpret_tensor(buf63, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf63  # reuse
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf75, (8, 64, 1024), (64, 1, 512), 0), out=buf76)
    buf77 = buf61; del buf61  # reuse
    buf78 = reinterpret_tensor(buf76, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf76  # reuse
    buf79 = buf59; del buf59  # reuse
    cpp_fused__softmax_23(c_void_p(buf78.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = buf75; del buf75  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (1024, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf80)
    del arg54_1
    buf81 = buf78; del buf78  # reuse
    cpp_fused__softmax_24(c_void_p(buf81.data_ptr()), c_void_p(buf79.data_ptr()))
    buf82 = reinterpret_tensor(buf73, (8, 1024, 64), (65536, 64, 1), 0); del buf73  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf80, (8, 1024, 64), (64, 512, 1), 0), out=buf82)
    buf83 = reinterpret_tensor(buf80, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf80  # reuse
    cpp_fused_clone_25(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf82, (1024, 512), (512, 1), 0); del buf82  # reuse
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (1024, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf84)
    del arg55_1
    buf85 = buf48; del buf48  # reuse
    buf86 = buf72; del buf72  # reuse
    buf87 = reinterpret_tensor(buf83, (1, 1024, 512), (524288, 512, 1), 0); del buf83  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_26(c_void_p(buf85.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del arg7_1
    buf88 = reinterpret_tensor(buf70, (1024, 2048), (2048, 1), 0); del buf70  # reuse
    # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (1024, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf88)
    del arg56_1
    buf89 = reinterpret_tensor(buf88, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf88  # reuse
    cpp_fused_relu_27(c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf87, (1024, 512), (512, 1), 0); del buf87  # reuse
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf89, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 512), (1, 2048), 0), out=buf90)
    del arg57_1
    buf91 = buf86; del buf86  # reuse
    buf92 = reinterpret_tensor(buf84, (1, 1024, 512), (524288, 512, 1), 0); del buf84  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_28(c_void_p(buf85.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg8_1
    buf93 = buf71; del buf71  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (1024, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf93)
    del arg58_1
    buf94 = buf66; del buf66  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (1024, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf94)
    del arg59_1
    buf95 = reinterpret_tensor(buf81, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf81  # reuse
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf93, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf94, (8, 64, 1024), (64, 1, 512), 0), out=buf95)
    buf96 = buf79; del buf79  # reuse
    buf97 = reinterpret_tensor(buf95, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf95  # reuse
    buf98 = buf77; del buf77  # reuse
    cpp_fused__softmax_29(c_void_p(buf97.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = buf94; del buf94  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (1024, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf99)
    del arg60_1
    buf100 = buf97; del buf97  # reuse
    cpp_fused__softmax_30(c_void_p(buf100.data_ptr()), c_void_p(buf98.data_ptr()))
    buf101 = reinterpret_tensor(buf92, (8, 1024, 64), (65536, 64, 1), 0); del buf92  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf100, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf99, (8, 1024, 64), (64, 512, 1), 0), out=buf101)
    buf102 = reinterpret_tensor(buf99, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf99  # reuse
    cpp_fused_clone_31(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf101, (1024, 512), (512, 1), 0); del buf101  # reuse
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (1024, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf103)
    del arg61_1
    buf104 = buf91; del buf91  # reuse
    buf105 = reinterpret_tensor(buf102, (1, 1024, 512), (524288, 512, 1), 0); del buf102  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_32(c_void_p(buf85.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg9_1
    buf106 = reinterpret_tensor(buf89, (1024, 2048), (2048, 1), 0); del buf89  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (1024, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 2048), (1, 512), 0), out=buf106)
    del arg62_1
    buf107 = reinterpret_tensor(buf106, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf106  # reuse
    cpp_fused_relu_33(c_void_p(buf107.data_ptr()))
    buf108 = reinterpret_tensor(buf105, (1024, 512), (512, 1), 0); del buf105  # reuse
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 512), (1, 2048), 0), out=buf108)
    del arg63_1
    buf109 = buf104; del buf104  # reuse
    buf110 = reinterpret_tensor(buf93, (1, 1024, 512), (524288, 512, 1), 0); del buf93  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_34(c_void_p(buf85.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg10_1
    buf111 = buf53; del buf53  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (1024, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf111)
    del arg64_1
    buf112 = buf74; del buf74  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (1024, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 512), (1, 512), 0), out=buf112)
    del arg65_1
    buf113 = reinterpret_tensor(buf100, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf100  # reuse
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf111, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf112, (8, 64, 1024), (64, 1, 512), 0), out=buf113)
    buf114 = buf98; del buf98  # reuse
    buf115 = reinterpret_tensor(buf113, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf113  # reuse
    buf116 = buf96; del buf96  # reuse
    cpp_fused__softmax_35(c_void_p(buf115.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg36_1
    buf117 = buf112; del buf112  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (1024, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf117)
    del arg66_1
    buf118 = buf115; del buf115  # reuse
    cpp_fused__softmax_36(c_void_p(buf118.data_ptr()), c_void_p(buf116.data_ptr()))
    buf119 = reinterpret_tensor(buf110, (8, 1024, 64), (65536, 64, 1), 0); del buf110  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf118, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf117, (8, 1024, 64), (64, 512, 1), 0), out=buf119)
    buf120 = reinterpret_tensor(buf117, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf117  # reuse
    cpp_fused_clone_37(c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf119, (1024, 512), (512, 1), 0); del buf119  # reuse
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (1024, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 512), (1, 512), 0), out=buf121)
    del arg67_1
    buf122 = reinterpret_tensor(buf103, (1, 1024, 512), (524288, 512, 1), 0); del buf103  # reuse
    buf123 = buf109; del buf109  # reuse
    buf124 = reinterpret_tensor(buf120, (1, 1024, 512), (524288, 512, 1), 0); del buf120  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_38(c_void_p(buf122.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    del arg11_1
    buf125 = reinterpret_tensor(buf107, (1024, 2048), (2048, 1), 0); del buf107  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (1024, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 2048), (1, 512), 0), out=buf125)
    del arg68_1
    buf126 = reinterpret_tensor(buf125, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf125  # reuse
    cpp_fused_relu_39(c_void_p(buf126.data_ptr()))
    buf127 = reinterpret_tensor(buf124, (1024, 512), (512, 1), 0); del buf124  # reuse
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 512), (1, 2048), 0), out=buf127)
    del arg69_1
    buf128 = buf123; del buf123  # reuse
    buf129 = buf122; del buf122  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_40(c_void_p(buf129.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg12_1
    buf130 = buf127; del buf127  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 512), (1, 512), 0), out=buf130)
    del arg76_1
    buf131 = reinterpret_tensor(buf118, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf118  # reuse
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf16, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf130, (8, 64, 1024), (64, 1, 512), 0), out=buf131)
    buf132 = buf116; del buf116  # reuse
    buf133 = reinterpret_tensor(buf131, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf131  # reuse
    buf134 = buf114; del buf114  # reuse
    cpp_fused__softmax_41(c_void_p(buf133.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()))
    buf135 = buf16; del buf16  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 512), (1, 512), 0), out=buf135)
    del arg77_1
    buf136 = buf133; del buf133  # reuse
    cpp_fused__softmax_42(c_void_p(buf136.data_ptr()), c_void_p(buf134.data_ptr()))
    buf137 = reinterpret_tensor(buf90, (8, 1024, 64), (65536, 64, 1), 0); del buf90  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf135, (8, 1024, 64), (64, 512, 1), 0), out=buf137)
    buf138 = reinterpret_tensor(buf85, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf85  # reuse
    cpp_fused_clone_43(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    buf139 = reinterpret_tensor(buf137, (1024, 512), (512, 1), 0); del buf137  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (1024, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 512), (1, 512), 0), out=buf139)
    del arg78_1
    buf140 = buf128; del buf128  # reuse
    buf141 = reinterpret_tensor(buf138, (1, 1024, 512), (524288, 512, 1), 0); del buf138  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_44(c_void_p(arg134_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    del arg15_1
    buf142 = reinterpret_tensor(buf126, (1024, 2048), (2048, 1), 0); del buf126  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf141, (1024, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf142)
    del arg79_1
    buf143 = reinterpret_tensor(buf142, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf142  # reuse
    cpp_fused_relu_45(c_void_p(buf143.data_ptr()))
    buf144 = reinterpret_tensor(buf141, (1024, 512), (512, 1), 0); del buf141  # reuse
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf143, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg80_1, (2048, 512), (1, 2048), 0), out=buf144)
    del arg80_1
    buf145 = reinterpret_tensor(buf13, (1, 1024, 512), (524288, 512, 1), 0); del buf13  # reuse
    buf146 = buf140; del buf140  # reuse
    buf147 = reinterpret_tensor(buf121, (1, 1024, 512), (524288, 512, 1), 0); del buf121  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_46(c_void_p(buf145.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del arg134_1
    del arg16_1
    del arg32_1
    buf148 = buf144; del buf144  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1024, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 512), (1, 512), 0), out=buf148)
    del arg81_1
    buf149 = buf139; del buf139  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1024, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 512), (1, 512), 0), out=buf149)
    del arg82_1
    buf150 = reinterpret_tensor(buf136, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf136  # reuse
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf148, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf149, (8, 64, 1024), (64, 1, 512), 0), out=buf150)
    buf151 = buf134; del buf134  # reuse
    buf152 = reinterpret_tensor(buf150, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf150  # reuse
    buf153 = buf152; del buf152  # reuse
    buf154 = buf132; del buf132  # reuse
    cpp_fused__softmax_47(c_void_p(buf153.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = buf148; del buf148  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1024, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 512), (1, 512), 0), out=buf155)
    del arg83_1
    buf156 = buf153; del buf153  # reuse
    cpp_fused__softmax_48(c_void_p(buf156.data_ptr()), c_void_p(buf154.data_ptr()))
    buf157 = reinterpret_tensor(buf147, (8, 1024, 64), (65536, 64, 1), 0); del buf147  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf156, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf155, (8, 1024, 64), (64, 512, 1), 0), out=buf157)
    buf158 = reinterpret_tensor(buf108, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf108  # reuse
    cpp_fused_clone_49(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf157, (1024, 512), (512, 1), 0); del buf157  # reuse
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (1024, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf159)
    del arg84_1
    buf160 = buf146; del buf146  # reuse
    buf161 = reinterpret_tensor(buf158, (1, 1024, 512), (524288, 512, 1), 0); del buf158  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_50(c_void_p(buf145.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg17_1
    buf162 = buf111; del buf111  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1024, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf162)
    del arg85_1
    buf163 = reinterpret_tensor(buf161, (1024, 512), (512, 1), 0); del buf161  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf163)
    del arg86_1
    buf164 = reinterpret_tensor(buf156, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf156  # reuse
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf162, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf163, (8, 64, 1024), (64, 1, 512), 0), out=buf164)
    buf165 = buf154; del buf154  # reuse
    buf166 = reinterpret_tensor(buf164, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf164  # reuse
    buf167 = buf151; del buf151  # reuse
    cpp_fused__softmax_51(c_void_p(buf166.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = buf162; del buf162  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf168)
    del arg87_1
    buf169 = buf166; del buf166  # reuse
    cpp_fused__softmax_52(c_void_p(buf169.data_ptr()), c_void_p(buf167.data_ptr()))
    buf170 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf168, (8, 1024, 64), (64, 512, 1), 0), out=buf170)
    buf171 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_53(c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf170, (1024, 512), (512, 1), 0); del buf170  # reuse
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (1024, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf172)
    del arg88_1
    buf173 = buf160; del buf160  # reuse
    buf174 = reinterpret_tensor(buf171, (1, 1024, 512), (524288, 512, 1), 0); del buf171  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_54(c_void_p(buf145.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg18_1
    buf175 = reinterpret_tensor(buf143, (1024, 2048), (2048, 1), 0); del buf143  # reuse
    # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (1024, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf175)
    del arg89_1
    buf176 = reinterpret_tensor(buf175, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf175  # reuse
    cpp_fused_relu_55(c_void_p(buf176.data_ptr()))
    buf177 = reinterpret_tensor(buf174, (1024, 512), (512, 1), 0); del buf174  # reuse
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf177)
    del arg90_1
    buf178 = buf173; del buf173  # reuse
    buf179 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_56(c_void_p(buf145.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    del arg19_1
    buf180 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (1024, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf180)
    del arg91_1
    buf181 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (1024, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 512), (1, 512), 0), out=buf181)
    del arg92_1
    buf182 = reinterpret_tensor(buf169, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf169  # reuse
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf180, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf181, (8, 64, 1024), (64, 1, 512), 0), out=buf182)
    buf183 = buf167; del buf167  # reuse
    buf184 = reinterpret_tensor(buf182, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf182  # reuse
    buf185 = buf184; del buf184  # reuse
    buf186 = buf165; del buf165  # reuse
    cpp_fused__softmax_57(c_void_p(buf185.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = buf180; del buf180  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (1024, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf187)
    del arg93_1
    buf188 = buf185; del buf185  # reuse
    cpp_fused__softmax_58(c_void_p(buf188.data_ptr()), c_void_p(buf186.data_ptr()))
    buf189 = reinterpret_tensor(buf179, (8, 1024, 64), (65536, 64, 1), 0); del buf179  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf187, (8, 1024, 64), (64, 512, 1), 0), out=buf189)
    buf190 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_59(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf189, (1024, 512), (512, 1), 0); del buf189  # reuse
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (1024, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 512), (1, 512), 0), out=buf191)
    del arg94_1
    buf192 = buf145; del buf145  # reuse
    buf193 = buf178; del buf178  # reuse
    buf194 = reinterpret_tensor(buf190, (1, 1024, 512), (524288, 512, 1), 0); del buf190  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_60(c_void_p(buf192.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg20_1
    buf195 = buf191; del buf191  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (1024, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 512), (1, 512), 0), out=buf195)
    del arg95_1
    buf196 = reinterpret_tensor(buf194, (1024, 512), (512, 1), 0); del buf194  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf196)
    del arg96_1
    buf197 = reinterpret_tensor(buf188, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf188  # reuse
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf196, (8, 64, 1024), (64, 1, 512), 0), out=buf197)
    buf198 = buf186; del buf186  # reuse
    buf199 = reinterpret_tensor(buf197, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf197  # reuse
    buf200 = buf183; del buf183  # reuse
    cpp_fused__softmax_61(c_void_p(buf199.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = buf195; del buf195  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 512), (1, 512), 0), out=buf201)
    del arg97_1
    buf202 = buf199; del buf199  # reuse
    cpp_fused__softmax_62(c_void_p(buf202.data_ptr()), c_void_p(buf200.data_ptr()))
    buf203 = reinterpret_tensor(buf177, (8, 1024, 64), (65536, 64, 1), 0); del buf177  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf202, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf201, (8, 1024, 64), (64, 512, 1), 0), out=buf203)
    buf204 = reinterpret_tensor(buf172, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf172  # reuse
    cpp_fused_clone_63(c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf203, (1024, 512), (512, 1), 0); del buf203  # reuse
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (1024, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf205)
    del arg98_1
    buf206 = buf193; del buf193  # reuse
    buf207 = reinterpret_tensor(buf204, (1, 1024, 512), (524288, 512, 1), 0); del buf204  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_64(c_void_p(buf192.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg21_1
    buf208 = reinterpret_tensor(buf176, (1024, 2048), (2048, 1), 0); del buf176  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (1024, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 2048), (1, 512), 0), out=buf208)
    del arg99_1
    buf209 = reinterpret_tensor(buf208, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf208  # reuse
    cpp_fused_relu_65(c_void_p(buf209.data_ptr()))
    buf210 = reinterpret_tensor(buf207, (1024, 512), (512, 1), 0); del buf207  # reuse
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 512), (1, 2048), 0), out=buf210)
    del arg100_1
    buf211 = buf206; del buf206  # reuse
    buf212 = reinterpret_tensor(buf159, (1, 1024, 512), (524288, 512, 1), 0); del buf159  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_66(c_void_p(buf192.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg22_1
    buf213 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (1024, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf213)
    del arg101_1
    buf214 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (1024, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf214)
    del arg102_1
    buf215 = reinterpret_tensor(buf202, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf202  # reuse
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf214, (8, 64, 1024), (64, 1, 512), 0), out=buf215)
    buf216 = buf200; del buf200  # reuse
    buf217 = reinterpret_tensor(buf215, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf215  # reuse
    buf218 = buf217; del buf217  # reuse
    buf219 = buf198; del buf198  # reuse
    cpp_fused__softmax_67(c_void_p(buf218.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = buf213; del buf213  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (1024, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), out=buf220)
    del arg103_1
    buf221 = buf218; del buf218  # reuse
    cpp_fused__softmax_68(c_void_p(buf221.data_ptr()), c_void_p(buf219.data_ptr()))
    buf222 = reinterpret_tensor(buf212, (8, 1024, 64), (65536, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf221, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf220, (8, 1024, 64), (64, 512, 1), 0), out=buf222)
    buf223 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_69(c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf222, (1024, 512), (512, 1), 0); del buf222  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (1024, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf224)
    del arg104_1
    buf225 = buf211; del buf211  # reuse
    buf226 = reinterpret_tensor(buf223, (1, 1024, 512), (524288, 512, 1), 0); del buf223  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_70(c_void_p(buf192.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg23_1
    buf227 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (1024, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), out=buf227)
    del arg105_1
    buf228 = reinterpret_tensor(buf226, (1024, 512), (512, 1), 0); del buf226  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf228)
    del arg106_1
    buf229 = reinterpret_tensor(buf221, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf221  # reuse
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf228, (8, 64, 1024), (64, 1, 512), 0), out=buf229)
    buf230 = buf219; del buf219  # reuse
    buf231 = reinterpret_tensor(buf229, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf229  # reuse
    buf232 = buf216; del buf216  # reuse
    cpp_fused__softmax_71(c_void_p(buf231.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = buf227; del buf227  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf233)
    del arg107_1
    buf234 = buf231; del buf231  # reuse
    cpp_fused__softmax_72(c_void_p(buf234.data_ptr()), c_void_p(buf232.data_ptr()))
    buf235 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf234, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf233, (8, 1024, 64), (64, 512, 1), 0), out=buf235)
    buf236 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_73(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf235, (1024, 512), (512, 1), 0); del buf235  # reuse
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (1024, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf237)
    del arg108_1
    buf238 = buf192; del buf192  # reuse
    buf239 = buf225; del buf225  # reuse
    buf240 = reinterpret_tensor(buf236, (1, 1024, 512), (524288, 512, 1), 0); del buf236  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_74(c_void_p(buf238.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg24_1
    buf241 = reinterpret_tensor(buf209, (1024, 2048), (2048, 1), 0); del buf209  # reuse
    # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (1024, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 2048), (1, 512), 0), out=buf241)
    del arg109_1
    buf242 = reinterpret_tensor(buf241, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf241  # reuse
    cpp_fused_relu_75(c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf240, (1024, 512), (512, 1), 0); del buf240  # reuse
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 512), (1, 2048), 0), out=buf243)
    del arg110_1
    buf244 = buf239; del buf239  # reuse
    buf245 = reinterpret_tensor(buf237, (1, 1024, 512), (524288, 512, 1), 0); del buf237  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_76(c_void_p(buf238.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg25_1
    buf246 = buf224; del buf224  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 512), (1, 512), 0), out=buf246)
    del arg111_1
    buf247 = buf210; del buf210  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf247)
    del arg112_1
    buf248 = reinterpret_tensor(buf234, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf234  # reuse
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf246, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf247, (8, 64, 1024), (64, 1, 512), 0), out=buf248)
    buf249 = buf232; del buf232  # reuse
    buf250 = reinterpret_tensor(buf248, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf248  # reuse
    buf251 = buf250; del buf250  # reuse
    buf252 = buf230; del buf230  # reuse
    cpp_fused__softmax_77(c_void_p(buf251.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf252.data_ptr()))
    buf253 = buf246; del buf246  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 512), (1, 512), 0), out=buf253)
    del arg113_1
    buf254 = buf251; del buf251  # reuse
    cpp_fused__softmax_78(c_void_p(buf254.data_ptr()), c_void_p(buf252.data_ptr()))
    buf255 = reinterpret_tensor(buf245, (8, 1024, 64), (65536, 64, 1), 0); del buf245  # reuse
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf253, (8, 1024, 64), (64, 512, 1), 0), out=buf255)
    buf256 = reinterpret_tensor(buf205, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf205  # reuse
    cpp_fused_clone_79(c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf255, (1024, 512), (512, 1), 0); del buf255  # reuse
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (1024, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf257)
    del arg114_1
    buf258 = buf244; del buf244  # reuse
    buf259 = reinterpret_tensor(buf256, (1, 1024, 512), (524288, 512, 1), 0); del buf256  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_80(c_void_p(buf238.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg26_1
    buf260 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (1024, 512), (512, 1), 0), reinterpret_tensor(arg115_1, (512, 512), (1, 512), 0), out=buf260)
    del arg115_1
    buf261 = reinterpret_tensor(buf259, (1024, 512), (512, 1), 0); del buf259  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 512), (1, 512), 0), out=buf261)
    del arg116_1
    buf262 = reinterpret_tensor(buf254, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf254  # reuse
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf260, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf261, (8, 64, 1024), (64, 1, 512), 0), out=buf262)
    buf263 = buf252; del buf252  # reuse
    buf264 = reinterpret_tensor(buf262, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf262  # reuse
    buf265 = buf249; del buf249  # reuse
    cpp_fused__softmax_81(c_void_p(buf264.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = buf260; del buf260  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), out=buf266)
    del arg117_1
    buf267 = buf264; del buf264  # reuse
    cpp_fused__softmax_82(c_void_p(buf267.data_ptr()), c_void_p(buf265.data_ptr()))
    buf268 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf267, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf266, (8, 1024, 64), (64, 512, 1), 0), out=buf268)
    buf269 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_83(c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf268, (1024, 512), (512, 1), 0); del buf268  # reuse
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (1024, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 512), (1, 512), 0), out=buf270)
    del arg118_1
    buf271 = buf258; del buf258  # reuse
    buf272 = reinterpret_tensor(buf269, (1, 1024, 512), (524288, 512, 1), 0); del buf269  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_84(c_void_p(buf238.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del arg27_1
    buf273 = reinterpret_tensor(buf242, (1024, 2048), (2048, 1), 0); del buf242  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (1024, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 2048), (1, 512), 0), out=buf273)
    del arg119_1
    buf274 = reinterpret_tensor(buf273, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf273  # reuse
    cpp_fused_relu_85(c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf272, (1024, 512), (512, 1), 0); del buf272  # reuse
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 512), (1, 2048), 0), out=buf275)
    del arg120_1
    buf276 = buf238; del buf238  # reuse
    buf277 = buf271; del buf271  # reuse
    buf278 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_86(c_void_p(buf276.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del arg28_1
    buf279 = buf275; del buf275  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf278, (1024, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf279)
    del arg121_1
    buf280 = buf270; del buf270  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf278, (1024, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf280)
    del arg122_1
    buf281 = reinterpret_tensor(buf267, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf267  # reuse
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf280, (8, 64, 1024), (64, 1, 512), 0), out=buf281)
    buf282 = buf265; del buf265  # reuse
    buf283 = reinterpret_tensor(buf281, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf281  # reuse
    buf284 = buf283; del buf283  # reuse
    buf285 = buf263; del buf263  # reuse
    cpp_fused__softmax_87(c_void_p(buf284.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg73_1
    buf286 = buf279; del buf279  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf278, (1024, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), out=buf286)
    del arg123_1
    buf287 = buf284; del buf284  # reuse
    cpp_fused__softmax_88(c_void_p(buf287.data_ptr()), c_void_p(buf285.data_ptr()))
    buf288 = reinterpret_tensor(buf278, (8, 1024, 64), (65536, 64, 1), 0); del buf278  # reuse
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf287, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf286, (8, 1024, 64), (64, 512, 1), 0), out=buf288)
    buf289 = reinterpret_tensor(buf257, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf257  # reuse
    cpp_fused_clone_89(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf288, (1024, 512), (512, 1), 0); del buf288  # reuse
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (1024, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf290)
    del arg124_1
    buf291 = buf277; del buf277  # reuse
    buf292 = reinterpret_tensor(buf289, (1, 1024, 512), (524288, 512, 1), 0); del buf289  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_90(c_void_p(buf276.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    del arg29_1
    buf293 = buf243; del buf243  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (1024, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 512), (1, 512), 0), out=buf293)
    del arg125_1
    buf294 = reinterpret_tensor(buf292, (1024, 512), (512, 1), 0); del buf292  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf294)
    del arg126_1
    buf295 = reinterpret_tensor(buf287, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf287  # reuse
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf293, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf294, (8, 64, 1024), (64, 1, 512), 0), out=buf295)
    buf296 = buf285; del buf285  # reuse
    buf297 = reinterpret_tensor(buf295, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf295  # reuse
    buf298 = buf282; del buf282  # reuse
    cpp_fused__softmax_91(c_void_p(buf297.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf296
    buf299 = buf293; del buf293  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 512), (1, 512), 0), out=buf299)
    del arg127_1
    buf300 = buf297; del buf297  # reuse
    cpp_fused__softmax_92(c_void_p(buf300.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf298
    buf301 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf299, (8, 1024, 64), (64, 512, 1), 0), out=buf301)
    del buf300
    buf302 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_93(c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf301, (1024, 512), (512, 1), 0); del buf301  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (1024, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 512), (1, 512), 0), out=buf303)
    del arg128_1
    buf304 = buf291; del buf291  # reuse
    buf305 = reinterpret_tensor(buf302, (1, 1024, 512), (524288, 512, 1), 0); del buf302  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_94(c_void_p(buf276.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del arg30_1
    buf306 = reinterpret_tensor(buf274, (1024, 2048), (2048, 1), 0); del buf274  # reuse
    # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (1024, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 2048), (1, 512), 0), out=buf306)
    del arg129_1
    buf307 = reinterpret_tensor(buf306, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf306  # reuse
    cpp_fused_relu_95(c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf305, (1024, 512), (512, 1), 0); del buf305  # reuse
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf308)
    del arg130_1
    del buf307
    buf309 = buf304; del buf304  # reuse
    buf310 = buf276; del buf276  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_96(c_void_p(buf310.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf309.data_ptr()))
    del arg31_1
    del buf290
    del buf303
    del buf308
    buf311 = empty((1024, 32128), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (1024, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 32128), (1, 512), 0), out=buf311)
    del arg131_1
    del buf310
    buf312 = reinterpret_tensor(buf309, (1024, 1), (1, 1024), 0); del buf309  # reuse
    buf313 = reinterpret_tensor(buf14, (1024, 1), (1, 1024), 0); del buf14  # reuse
    buf314 = empty((), device='cpu', dtype=torch.float32)
    buf315 = empty((), device='cpu', dtype=torch.int64)
    buf316 = buf314; del buf314  # reuse
    cpp_fused__log_softmax_nll_loss_forward_97(c_void_p(buf316.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()))
    del arg133_1
    return (buf316, reinterpret_tensor(buf311, (1, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf3, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf9, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf130, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf135, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf149, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf155, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf163, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf168, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf181, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf187, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf196, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf201, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf214, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf220, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf228, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf233, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf247, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf253, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf261, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf266, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf280, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf286, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf294, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf299, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), buf129, )


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
    arg132_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    arg133_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    arg134_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5Small', benchmark_compiled_module)
