
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


cpp_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 512);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp11 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 512);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        auto tmp12 = static_cast<float>(768.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp10 - tmp14;
                        auto tmp16 = tmp15 * tmp15;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp6 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = out_ptr1[static_cast<long>(x0)];
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = decltype(tmp1)(tmp1 + 50265);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 50265L), "index out of bounds: 0 <= tmp4 < 50265L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp4)));
                    auto tmp7 = decltype(tmp6)(tmp6 + 512);
                    auto tmp8 = tmp6 < 0;
                    auto tmp9 = tmp8 ? tmp7 : tmp6;
                    TORCH_CHECK((0 <= tmp9) & (tmp9 < 512L), "index out of bounds: 0 <= tmp9 < 512L")
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp9)));
                    auto tmp11 = tmp5 + tmp10;
                    auto tmp13 = static_cast<float>(768.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp18 = tmp17 / tmp13;
                    auto tmp19 = static_cast<float>(1e-07);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = std::sqrt(tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp16 / tmp22;
                    auto tmp24 = tmp0 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp26.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_33 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_45 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(8.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp3);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<bool>(0);
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = tmp1 ? tmp2 : tmp0;
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = std::exp(tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp6;
                        tmp_acc0 = tmp_acc0 + tmp6;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_pow_sqrt_sub_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = static_cast<float>(768.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp2 - tmp6;
                        auto tmp8 = tmp7 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp10 = tmp9 / tmp5;
                    auto tmp11 = static_cast<float>(1e-07);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp8 / tmp14;
                    auto tmp16 = tmp0 * tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = out_ptr1[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr2[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = out_ptr4[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp11 = out_ptr1[static_cast<long>(0L)];
        auto tmp13 = out_ptr2[static_cast<long>(0L)];
        auto tmp22 = in_ptr2[static_cast<long>(0L)];
        auto tmp31 = out_ptr4[static_cast<long>(0L)];
        auto tmp33 = out_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(512);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        auto tmp7 = decltype(tmp6)(tmp6 + 512);
        auto tmp8 = tmp6 < 0;
        auto tmp9 = tmp8 ? tmp7 : tmp6;
        TORCH_CHECK((0 <= tmp9) & (tmp9 < 512L), "index out of bounds: 0 <= tmp9 < 512L")
        auto tmp10 = out_ptr0[static_cast<long>(tmp9)];
        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
        auto tmp14 = std::log(tmp13);
        auto tmp15 = decltype(tmp12)(tmp12 - tmp14);
        auto tmp16 = decltype(tmp15)(-tmp15);
        auto tmp17 = static_cast<float>(0.0);
        auto tmp18 = tmp5 ? tmp16 : tmp17;
        auto tmp19 = c10::convert<long>(tmp5);
        auto tmp20 = c10::convert<float>(tmp19);
        auto tmp21 = tmp18 / tmp20;
        auto tmp23 = max_propagate_nan(tmp22, tmp1);
        auto tmp24 = min_propagate_nan(tmp23, tmp3);
        auto tmp25 = tmp24 != tmp3;
        auto tmp26 = tmp25 ? tmp24 : tmp1;
        auto tmp27 = decltype(tmp26)(tmp26 + 512);
        auto tmp28 = tmp26 < 0;
        auto tmp29 = tmp28 ? tmp27 : tmp26;
        TORCH_CHECK((0 <= tmp29) & (tmp29 < 512L), "index out of bounds: 0 <= tmp29 < 512L")
        auto tmp30 = out_ptr3[static_cast<long>(tmp29)];
        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
        auto tmp34 = std::log(tmp33);
        auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
        auto tmp36 = decltype(tmp35)(-tmp35);
        auto tmp37 = tmp25 ? tmp36 : tmp17;
        auto tmp38 = c10::convert<long>(tmp25);
        auto tmp39 = c10::convert<float>(tmp38);
        auto tmp40 = tmp37 / tmp39;
        auto tmp41 = decltype(tmp21)(tmp21 + tmp40);
        auto tmp42 = static_cast<float>(2.0);
        auto tmp43 = tmp41 / tmp42;
        in_out_ptr0[static_cast<long>(0L)] = tmp43;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, ), (1, ))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (50265, 768), (768, 1))
    assert_size_stride(arg75_1, (512, 768), (768, 1))
    assert_size_stride(arg76_1, (2304, 768), (768, 1))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (2304, 768), (768, 1))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (3072, 768), (768, 1))
    assert_size_stride(arg87_1, (3072, ), (1, ))
    assert_size_stride(arg88_1, (768, 3072), (3072, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (2304, 768), (768, 1))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (3072, 768), (768, 1))
    assert_size_stride(arg94_1, (3072, ), (1, ))
    assert_size_stride(arg95_1, (768, 3072), (3072, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (2304, 768), (768, 1))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (3072, 768), (768, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (768, 3072), (3072, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (2304, 768), (768, 1))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (3072, 768), (768, 1))
    assert_size_stride(arg108_1, (3072, ), (1, ))
    assert_size_stride(arg109_1, (768, 3072), (3072, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (2304, 768), (768, 1))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (3072, 768), (768, 1))
    assert_size_stride(arg115_1, (3072, ), (1, ))
    assert_size_stride(arg116_1, (768, 3072), (3072, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (2304, 768), (768, 1))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (2304, 768), (768, 1))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (768, 3072), (3072, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (2304, 768), (768, 1))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (3072, 768), (768, 1))
    assert_size_stride(arg136_1, (3072, ), (1, ))
    assert_size_stride(arg137_1, (768, 3072), (3072, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (2304, 768), (768, 1))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (2304, 768), (768, 1))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (3072, 768), (768, 1))
    assert_size_stride(arg150_1, (3072, ), (1, ))
    assert_size_stride(arg151_1, (768, 3072), (3072, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (2304, 768), (768, 1))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (3072, 768), (768, 1))
    assert_size_stride(arg157_1, (3072, ), (1, ))
    assert_size_stride(arg158_1, (768, 3072), (3072, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (2, 768), (768, 1))
    assert_size_stride(arg161_1, (2, ), (1, ))
    assert_size_stride(arg162_1, (1, 512), (512, 1))
    assert_size_stride(arg163_1, (1, 512), (512, 1))
    assert_size_stride(arg164_1, (1, ), (1, ))
    assert_size_stride(arg165_1, (1, ), (1, ))
    buf0 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0(c_void_p(arg163_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()))
    del arg0_1
    del arg162_1
    del arg163_1
    del arg1_1
    del arg74_1
    del arg75_1
    buf3 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf2, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 2304), (1, 768), 0), out=buf3)
    del arg76_1
    buf4 = empty((1, 12, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_1(c_void_p(buf3.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg2_1
    buf5 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf3, (12, 64, 512), (192, 1, 2304), 64), out=buf5)
    buf6 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf7 = reinterpret_tensor(buf5, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf5  # reuse
    buf8 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf9 = buf7; del buf7  # reuse
    buf10 = buf4; del buf4  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_2(c_void_p(buf9.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg3_1
    buf11 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf9, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf10, (12, 512, 64), (32768, 64, 1), 0), out=buf11)
    buf12 = reinterpret_tensor(buf10, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf10  # reuse
    cpp_fused_clone_3(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = reinterpret_tensor(buf11, (512, 768), (768, 1), 0); del buf11  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf12, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf13)
    del arg77_1
    del arg78_1
    buf14 = buf1; del buf1  # reuse
    buf15 = buf0; del buf0  # reuse
    buf16 = reinterpret_tensor(buf13, (1, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_4(c_void_p(buf16.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg4_1
    del arg5_1
    buf17 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf16, (512, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf17)
    del arg79_1
    del arg80_1
    buf18 = reinterpret_tensor(buf17, (1, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
    cpp_fused_gelu_5(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf2, (512, 768), (768, 1), 0); del buf2  # reuse
    # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf19)
    del arg81_1
    del arg82_1
    buf20 = buf15; del buf15  # reuse
    buf21 = buf14; del buf14  # reuse
    buf22 = buf16; del buf16  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_6(c_void_p(buf22.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg6_1
    del arg7_1
    buf23 = buf3; del buf3  # reuse
    # Source Nodes: [qp_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 2304), (1, 768), 0), out=buf23)
    del arg83_1
    buf24 = reinterpret_tensor(buf19, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf19  # reuse
    cpp_fused_add_div_sqrt_7(c_void_p(buf23.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf24.data_ptr()))
    del arg8_1
    buf25 = reinterpret_tensor(buf9, (12, 512, 512), (262144, 512, 1), 0); del buf9  # reuse
    # Source Nodes: [attention_scores_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf23, (12, 64, 512), (192, 1, 2304), 64), out=buf25)
    buf26 = buf8; del buf8  # reuse
    buf27 = reinterpret_tensor(buf25, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf25  # reuse
    buf28 = buf6; del buf6  # reuse
    buf29 = buf27; del buf27  # reuse
    buf30 = buf24; del buf24  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_8(c_void_p(buf29.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg9_1
    buf31 = reinterpret_tensor(buf12, (12, 512, 64), (32768, 64, 1), 0); del buf12  # reuse
    # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf29, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf30, (12, 512, 64), (32768, 64, 1), 0), out=buf31)
    buf32 = reinterpret_tensor(buf30, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf30  # reuse
    cpp_fused_clone_9(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf31, (512, 768), (768, 1), 0); del buf31  # reuse
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf32, (512, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
    del arg84_1
    del arg85_1
    buf34 = buf21; del buf21  # reuse
    buf35 = buf20; del buf20  # reuse
    buf36 = buf22; del buf22  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_10(c_void_p(buf36.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del arg10_1
    del arg11_1
    buf37 = reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0); del buf18  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf36, (512, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf37)
    del arg86_1
    del arg87_1
    buf38 = reinterpret_tensor(buf37, (1, 512, 3072), (1572864, 3072, 1), 0); del buf37  # reuse
    cpp_fused_gelu_11(c_void_p(buf38.data_ptr()))
    buf39 = buf33; del buf33  # reuse
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg88_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf39)
    del arg88_1
    del arg89_1
    buf40 = buf35; del buf35  # reuse
    buf41 = buf34; del buf34  # reuse
    buf42 = buf36; del buf36  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_12(c_void_p(buf42.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg12_1
    del arg13_1
    buf43 = buf23; del buf23  # reuse
    # Source Nodes: [qp_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (512, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 2304), (1, 768), 0), out=buf43)
    del arg90_1
    buf44 = reinterpret_tensor(buf39, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf39  # reuse
    cpp_fused_add_div_sqrt_13(c_void_p(buf43.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg14_1
    buf45 = reinterpret_tensor(buf29, (12, 512, 512), (262144, 512, 1), 0); del buf29  # reuse
    # Source Nodes: [attention_scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf43, (12, 64, 512), (192, 1, 2304), 64), out=buf45)
    buf46 = buf28; del buf28  # reuse
    buf47 = reinterpret_tensor(buf45, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf45  # reuse
    buf48 = buf26; del buf26  # reuse
    buf49 = buf47; del buf47  # reuse
    buf50 = buf44; del buf44  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_14(c_void_p(buf49.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg15_1
    buf51 = reinterpret_tensor(buf32, (12, 512, 64), (32768, 64, 1), 0); del buf32  # reuse
    # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf50, (12, 512, 64), (32768, 64, 1), 0), out=buf51)
    buf52 = reinterpret_tensor(buf50, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf50  # reuse
    cpp_fused_clone_15(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf51, (512, 768), (768, 1), 0); del buf51  # reuse
    # Source Nodes: [hidden_states_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf52, (512, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
    del arg91_1
    del arg92_1
    buf54 = buf41; del buf41  # reuse
    buf55 = buf40; del buf40  # reuse
    buf56 = buf42; del buf42  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_16(c_void_p(buf56.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg16_1
    del arg17_1
    buf57 = reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0); del buf38  # reuse
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf56, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf57)
    del arg93_1
    del arg94_1
    buf58 = reinterpret_tensor(buf57, (1, 512, 3072), (1572864, 3072, 1), 0); del buf57  # reuse
    cpp_fused_gelu_17(c_void_p(buf58.data_ptr()))
    buf59 = buf53; del buf53  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf58, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg95_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf59)
    del arg95_1
    del arg96_1
    buf60 = buf55; del buf55  # reuse
    buf61 = buf54; del buf54  # reuse
    buf62 = buf56; del buf56  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_18(c_void_p(buf62.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg18_1
    del arg19_1
    buf63 = buf43; del buf43  # reuse
    # Source Nodes: [qp_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (512, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 2304), (1, 768), 0), out=buf63)
    del arg97_1
    buf64 = reinterpret_tensor(buf59, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf59  # reuse
    cpp_fused_add_div_sqrt_19(c_void_p(buf63.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg20_1
    buf65 = reinterpret_tensor(buf49, (12, 512, 512), (262144, 512, 1), 0); del buf49  # reuse
    # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf64, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf63, (12, 64, 512), (192, 1, 2304), 64), out=buf65)
    buf66 = buf48; del buf48  # reuse
    buf67 = reinterpret_tensor(buf65, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf65  # reuse
    buf68 = buf46; del buf46  # reuse
    buf69 = buf67; del buf67  # reuse
    buf70 = buf64; del buf64  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_20(c_void_p(buf69.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg21_1
    buf71 = reinterpret_tensor(buf52, (12, 512, 64), (32768, 64, 1), 0); del buf52  # reuse
    # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf69, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf70, (12, 512, 64), (32768, 64, 1), 0), out=buf71)
    buf72 = reinterpret_tensor(buf70, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf70  # reuse
    cpp_fused_clone_21(c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf71, (512, 768), (768, 1), 0); del buf71  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf72, (512, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del arg98_1
    del arg99_1
    buf74 = buf61; del buf61  # reuse
    buf75 = buf60; del buf60  # reuse
    buf76 = buf62; del buf62  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_22(c_void_p(buf76.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg22_1
    del arg23_1
    buf77 = reinterpret_tensor(buf58, (512, 3072), (3072, 1), 0); del buf58  # reuse
    # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf76, (512, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf77)
    del arg100_1
    del arg101_1
    buf78 = reinterpret_tensor(buf77, (1, 512, 3072), (1572864, 3072, 1), 0); del buf77  # reuse
    cpp_fused_gelu_23(c_void_p(buf78.data_ptr()))
    buf79 = buf73; del buf73  # reuse
    # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf78, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf79)
    del arg102_1
    del arg103_1
    buf80 = buf75; del buf75  # reuse
    buf81 = buf74; del buf74  # reuse
    buf82 = buf76; del buf76  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_24(c_void_p(buf82.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del arg24_1
    del arg25_1
    buf83 = buf63; del buf63  # reuse
    # Source Nodes: [qp_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (512, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 2304), (1, 768), 0), out=buf83)
    del arg104_1
    buf84 = reinterpret_tensor(buf79, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf79  # reuse
    cpp_fused_add_div_sqrt_25(c_void_p(buf83.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg26_1
    buf85 = reinterpret_tensor(buf69, (12, 512, 512), (262144, 512, 1), 0); del buf69  # reuse
    # Source Nodes: [attention_scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf83, (12, 64, 512), (192, 1, 2304), 64), out=buf85)
    buf86 = buf68; del buf68  # reuse
    buf87 = reinterpret_tensor(buf85, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf85  # reuse
    buf88 = buf66; del buf66  # reuse
    buf89 = buf87; del buf87  # reuse
    buf90 = buf84; del buf84  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_26(c_void_p(buf89.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg27_1
    buf91 = reinterpret_tensor(buf72, (12, 512, 64), (32768, 64, 1), 0); del buf72  # reuse
    # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf89, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf90, (12, 512, 64), (32768, 64, 1), 0), out=buf91)
    buf92 = reinterpret_tensor(buf90, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf90  # reuse
    cpp_fused_clone_27(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf91, (512, 768), (768, 1), 0); del buf91  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf92, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf93)
    del arg105_1
    del arg106_1
    buf94 = buf81; del buf81  # reuse
    buf95 = buf80; del buf80  # reuse
    buf96 = buf82; del buf82  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_28(c_void_p(buf96.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg28_1
    del arg29_1
    buf97 = reinterpret_tensor(buf78, (512, 3072), (3072, 1), 0); del buf78  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf96, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg107_1
    del arg108_1
    buf98 = reinterpret_tensor(buf97, (1, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
    cpp_fused_gelu_29(c_void_p(buf98.data_ptr()))
    buf99 = buf93; del buf93  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg109_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf99)
    del arg109_1
    del arg110_1
    buf100 = buf95; del buf95  # reuse
    buf101 = buf94; del buf94  # reuse
    buf102 = buf96; del buf96  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_30(c_void_p(buf102.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg30_1
    del arg31_1
    buf103 = buf83; del buf83  # reuse
    # Source Nodes: [qp_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (512, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 2304), (1, 768), 0), out=buf103)
    del arg111_1
    buf104 = reinterpret_tensor(buf99, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf99  # reuse
    cpp_fused_add_div_sqrt_31(c_void_p(buf103.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf104.data_ptr()))
    del arg32_1
    buf105 = reinterpret_tensor(buf89, (12, 512, 512), (262144, 512, 1), 0); del buf89  # reuse
    # Source Nodes: [attention_scores_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf104, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf103, (12, 64, 512), (192, 1, 2304), 64), out=buf105)
    buf106 = buf88; del buf88  # reuse
    buf107 = reinterpret_tensor(buf105, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf105  # reuse
    buf108 = buf86; del buf86  # reuse
    buf109 = buf107; del buf107  # reuse
    buf110 = buf104; del buf104  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_32(c_void_p(buf109.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg33_1
    buf111 = reinterpret_tensor(buf92, (12, 512, 64), (32768, 64, 1), 0); del buf92  # reuse
    # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf110, (12, 512, 64), (32768, 64, 1), 0), out=buf111)
    buf112 = reinterpret_tensor(buf110, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf110  # reuse
    cpp_fused_clone_33(c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    buf113 = reinterpret_tensor(buf111, (512, 768), (768, 1), 0); del buf111  # reuse
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf112, (512, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf113)
    del arg112_1
    del arg113_1
    buf114 = buf101; del buf101  # reuse
    buf115 = buf100; del buf100  # reuse
    buf116 = buf102; del buf102  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_34(c_void_p(buf116.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg34_1
    del arg35_1
    buf117 = reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf116, (512, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf117)
    del arg114_1
    del arg115_1
    buf118 = reinterpret_tensor(buf117, (1, 512, 3072), (1572864, 3072, 1), 0); del buf117  # reuse
    cpp_fused_gelu_35(c_void_p(buf118.data_ptr()))
    buf119 = buf113; del buf113  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf118, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg116_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf119)
    del arg116_1
    del arg117_1
    buf120 = buf115; del buf115  # reuse
    buf121 = buf114; del buf114  # reuse
    buf122 = buf116; del buf116  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_36(c_void_p(buf122.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg36_1
    del arg37_1
    buf123 = buf103; del buf103  # reuse
    # Source Nodes: [qp_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 2304), (1, 768), 0), out=buf123)
    del arg118_1
    buf124 = reinterpret_tensor(buf119, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf119  # reuse
    cpp_fused_add_div_sqrt_37(c_void_p(buf123.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf124.data_ptr()))
    del arg38_1
    buf125 = reinterpret_tensor(buf109, (12, 512, 512), (262144, 512, 1), 0); del buf109  # reuse
    # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf124, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf123, (12, 64, 512), (192, 1, 2304), 64), out=buf125)
    buf126 = buf108; del buf108  # reuse
    buf127 = reinterpret_tensor(buf125, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf125  # reuse
    buf128 = buf106; del buf106  # reuse
    buf129 = buf127; del buf127  # reuse
    buf130 = buf124; del buf124  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_38(c_void_p(buf129.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg39_1
    buf131 = reinterpret_tensor(buf112, (12, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf129, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf130, (12, 512, 64), (32768, 64, 1), 0), out=buf131)
    buf132 = reinterpret_tensor(buf130, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf130  # reuse
    cpp_fused_clone_39(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf132, (512, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del arg119_1
    del arg120_1
    buf134 = buf121; del buf121  # reuse
    buf135 = buf120; del buf120  # reuse
    buf136 = buf122; del buf122  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_40(c_void_p(buf136.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg40_1
    del arg41_1
    buf137 = reinterpret_tensor(buf118, (512, 3072), (3072, 1), 0); del buf118  # reuse
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf136, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf137)
    del arg121_1
    del arg122_1
    buf138 = reinterpret_tensor(buf137, (1, 512, 3072), (1572864, 3072, 1), 0); del buf137  # reuse
    cpp_fused_gelu_41(c_void_p(buf138.data_ptr()))
    buf139 = buf133; del buf133  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf138, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf139)
    del arg123_1
    del arg124_1
    buf140 = buf135; del buf135  # reuse
    buf141 = buf134; del buf134  # reuse
    buf142 = buf136; del buf136  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_42(c_void_p(buf142.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    del arg42_1
    del arg43_1
    buf143 = buf123; del buf123  # reuse
    # Source Nodes: [qp_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (512, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 2304), (1, 768), 0), out=buf143)
    del arg125_1
    buf144 = reinterpret_tensor(buf139, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf139  # reuse
    cpp_fused_add_div_sqrt_43(c_void_p(buf143.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg44_1
    buf145 = reinterpret_tensor(buf129, (12, 512, 512), (262144, 512, 1), 0); del buf129  # reuse
    # Source Nodes: [attention_scores_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf143, (12, 64, 512), (192, 1, 2304), 64), out=buf145)
    buf146 = buf128; del buf128  # reuse
    buf147 = reinterpret_tensor(buf145, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf145  # reuse
    buf148 = buf126; del buf126  # reuse
    buf149 = buf147; del buf147  # reuse
    buf150 = buf144; del buf144  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_44(c_void_p(buf149.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    del arg45_1
    buf151 = reinterpret_tensor(buf132, (12, 512, 64), (32768, 64, 1), 0); del buf132  # reuse
    # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf150, (12, 512, 64), (32768, 64, 1), 0), out=buf151)
    buf152 = reinterpret_tensor(buf150, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf150  # reuse
    cpp_fused_clone_45(c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = reinterpret_tensor(buf151, (512, 768), (768, 1), 0); del buf151  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf152, (512, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
    del arg126_1
    del arg127_1
    buf154 = buf141; del buf141  # reuse
    buf155 = buf140; del buf140  # reuse
    buf156 = buf142; del buf142  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_46(c_void_p(buf156.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg46_1
    del arg47_1
    buf157 = reinterpret_tensor(buf138, (512, 3072), (3072, 1), 0); del buf138  # reuse
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf156, (512, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf157)
    del arg128_1
    del arg129_1
    buf158 = reinterpret_tensor(buf157, (1, 512, 3072), (1572864, 3072, 1), 0); del buf157  # reuse
    cpp_fused_gelu_47(c_void_p(buf158.data_ptr()))
    buf159 = buf153; del buf153  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf158, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf159)
    del arg130_1
    del arg131_1
    buf160 = buf155; del buf155  # reuse
    buf161 = buf154; del buf154  # reuse
    buf162 = buf156; del buf156  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_48(c_void_p(buf162.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg48_1
    del arg49_1
    buf163 = buf143; del buf143  # reuse
    # Source Nodes: [qp_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf162, (512, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 2304), (1, 768), 0), out=buf163)
    del arg132_1
    buf164 = reinterpret_tensor(buf159, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf159  # reuse
    cpp_fused_add_div_sqrt_49(c_void_p(buf163.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf164.data_ptr()))
    del arg50_1
    buf165 = reinterpret_tensor(buf149, (12, 512, 512), (262144, 512, 1), 0); del buf149  # reuse
    # Source Nodes: [attention_scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf164, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf163, (12, 64, 512), (192, 1, 2304), 64), out=buf165)
    buf166 = buf148; del buf148  # reuse
    buf167 = reinterpret_tensor(buf165, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf165  # reuse
    buf168 = buf146; del buf146  # reuse
    buf169 = buf167; del buf167  # reuse
    buf170 = buf164; del buf164  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_50(c_void_p(buf169.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del arg51_1
    buf171 = reinterpret_tensor(buf152, (12, 512, 64), (32768, 64, 1), 0); del buf152  # reuse
    # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf170, (12, 512, 64), (32768, 64, 1), 0), out=buf171)
    buf172 = reinterpret_tensor(buf170, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf170  # reuse
    cpp_fused_clone_51(c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
    # Source Nodes: [hidden_states_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf172, (512, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del arg133_1
    del arg134_1
    buf174 = buf161; del buf161  # reuse
    buf175 = buf160; del buf160  # reuse
    buf176 = buf162; del buf162  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_52(c_void_p(buf176.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del arg52_1
    del arg53_1
    buf177 = reinterpret_tensor(buf158, (512, 3072), (3072, 1), 0); del buf158  # reuse
    # Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf176, (512, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf177)
    del arg135_1
    del arg136_1
    buf178 = reinterpret_tensor(buf177, (1, 512, 3072), (1572864, 3072, 1), 0); del buf177  # reuse
    cpp_fused_gelu_53(c_void_p(buf178.data_ptr()))
    buf179 = buf173; del buf173  # reuse
    # Source Nodes: [hidden_states_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf178, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg137_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf179)
    del arg137_1
    del arg138_1
    buf180 = buf175; del buf175  # reuse
    buf181 = buf174; del buf174  # reuse
    buf182 = buf176; del buf176  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_54(c_void_p(buf182.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg54_1
    del arg55_1
    buf183 = buf163; del buf163  # reuse
    # Source Nodes: [qp_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 2304), (1, 768), 0), out=buf183)
    del arg139_1
    buf184 = reinterpret_tensor(buf179, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf179  # reuse
    cpp_fused_add_div_sqrt_55(c_void_p(buf183.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg56_1
    buf185 = reinterpret_tensor(buf169, (12, 512, 512), (262144, 512, 1), 0); del buf169  # reuse
    # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf183, (12, 64, 512), (192, 1, 2304), 64), out=buf185)
    buf186 = buf168; del buf168  # reuse
    buf187 = reinterpret_tensor(buf185, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf185  # reuse
    buf188 = buf166; del buf166  # reuse
    buf189 = buf187; del buf187  # reuse
    buf190 = buf184; del buf184  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_56(c_void_p(buf189.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()))
    del arg57_1
    buf191 = reinterpret_tensor(buf172, (12, 512, 64), (32768, 64, 1), 0); del buf172  # reuse
    # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf189, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf190, (12, 512, 64), (32768, 64, 1), 0), out=buf191)
    buf192 = reinterpret_tensor(buf190, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf190  # reuse
    cpp_fused_clone_57(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = reinterpret_tensor(buf191, (512, 768), (768, 1), 0); del buf191  # reuse
    # Source Nodes: [hidden_states_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf192, (512, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf193)
    del arg140_1
    del arg141_1
    buf194 = buf181; del buf181  # reuse
    buf195 = buf180; del buf180  # reuse
    buf196 = buf182; del buf182  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_58(c_void_p(buf196.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg58_1
    del arg59_1
    buf197 = reinterpret_tensor(buf178, (512, 3072), (3072, 1), 0); del buf178  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf196, (512, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf197)
    del arg142_1
    del arg143_1
    buf198 = reinterpret_tensor(buf197, (1, 512, 3072), (1572864, 3072, 1), 0); del buf197  # reuse
    cpp_fused_gelu_59(c_void_p(buf198.data_ptr()))
    buf199 = buf193; del buf193  # reuse
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf199)
    del arg144_1
    del arg145_1
    buf200 = buf195; del buf195  # reuse
    buf201 = buf194; del buf194  # reuse
    buf202 = buf196; del buf196  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_60(c_void_p(buf202.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del arg60_1
    del arg61_1
    buf203 = buf183; del buf183  # reuse
    # Source Nodes: [qp_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (512, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 2304), (1, 768), 0), out=buf203)
    del arg146_1
    buf204 = reinterpret_tensor(buf199, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf199  # reuse
    cpp_fused_add_div_sqrt_61(c_void_p(buf203.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf204.data_ptr()))
    del arg62_1
    buf205 = reinterpret_tensor(buf189, (12, 512, 512), (262144, 512, 1), 0); del buf189  # reuse
    # Source Nodes: [attention_scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf204, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf203, (12, 64, 512), (192, 1, 2304), 64), out=buf205)
    buf206 = buf188; del buf188  # reuse
    buf207 = reinterpret_tensor(buf205, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf205  # reuse
    buf208 = buf186; del buf186  # reuse
    buf209 = buf207; del buf207  # reuse
    buf210 = buf204; del buf204  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_62(c_void_p(buf209.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg63_1
    buf211 = reinterpret_tensor(buf192, (12, 512, 64), (32768, 64, 1), 0); del buf192  # reuse
    # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf209, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf210, (12, 512, 64), (32768, 64, 1), 0), out=buf211)
    buf212 = reinterpret_tensor(buf210, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf210  # reuse
    cpp_fused_clone_63(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = reinterpret_tensor(buf211, (512, 768), (768, 1), 0); del buf211  # reuse
    # Source Nodes: [hidden_states_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf212, (512, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf213)
    del arg147_1
    del arg148_1
    buf214 = buf201; del buf201  # reuse
    buf215 = buf200; del buf200  # reuse
    buf216 = buf202; del buf202  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_64(c_void_p(buf216.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del arg64_1
    del arg65_1
    buf217 = reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0); del buf198  # reuse
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf216, (512, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf217)
    del arg149_1
    del arg150_1
    buf218 = reinterpret_tensor(buf217, (1, 512, 3072), (1572864, 3072, 1), 0); del buf217  # reuse
    cpp_fused_gelu_65(c_void_p(buf218.data_ptr()))
    buf219 = buf213; del buf213  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf218, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg151_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf219)
    del arg151_1
    del arg152_1
    buf220 = buf215; del buf215  # reuse
    buf221 = buf214; del buf214  # reuse
    buf222 = buf216; del buf216  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_66(c_void_p(buf222.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del arg66_1
    del arg67_1
    buf223 = buf203; del buf203  # reuse
    # Source Nodes: [qp_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 2304), (1, 768), 0), out=buf223)
    del arg153_1
    buf224 = reinterpret_tensor(buf219, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf219  # reuse
    cpp_fused_add_div_sqrt_67(c_void_p(buf223.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf224.data_ptr()))
    del arg68_1
    buf225 = reinterpret_tensor(buf209, (12, 512, 512), (262144, 512, 1), 0); del buf209  # reuse
    # Source Nodes: [attention_scores_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf224, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf223, (12, 64, 512), (192, 1, 2304), 64), out=buf225)
    buf226 = buf208; del buf208  # reuse
    buf227 = reinterpret_tensor(buf225, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf225  # reuse
    buf228 = buf206; del buf206  # reuse
    buf229 = buf227; del buf227  # reuse
    buf230 = buf224; del buf224  # reuse
    cpp_fused__softmax_add_bitwise_not_lift_fresh_masked_fill_68(c_void_p(buf229.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    del arg69_1
    del buf223
    del buf226
    del buf228
    buf231 = reinterpret_tensor(buf212, (12, 512, 64), (32768, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf230, (12, 512, 64), (32768, 64, 1), 0), out=buf231)
    del buf229
    buf232 = reinterpret_tensor(buf230, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf230  # reuse
    cpp_fused_clone_69(c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf231, (512, 768), (768, 1), 0); del buf231  # reuse
    # Source Nodes: [hidden_states_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf232, (512, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf233)
    del arg154_1
    del arg155_1
    del buf232
    buf234 = buf221; del buf221  # reuse
    buf235 = buf220; del buf220  # reuse
    buf236 = buf222; del buf222  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_70(c_void_p(buf236.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del arg70_1
    del arg71_1
    buf237 = reinterpret_tensor(buf218, (512, 3072), (3072, 1), 0); del buf218  # reuse
    # Source Nodes: [hidden_states_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf236, (512, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf237)
    del arg156_1
    del arg157_1
    buf238 = reinterpret_tensor(buf237, (1, 512, 3072), (1572864, 3072, 1), 0); del buf237  # reuse
    cpp_fused_gelu_71(c_void_p(buf238.data_ptr()))
    buf239 = buf233; del buf233  # reuse
    # Source Nodes: [hidden_states_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf238, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg158_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf239)
    del arg158_1
    del arg159_1
    del buf238
    buf240 = buf235; del buf235  # reuse
    buf241 = buf234; del buf234  # reuse
    buf242 = buf236; del buf236  # reuse
    cpp_fused_add_div_mean_mul_pow_sqrt_sub_72(c_void_p(buf242.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg72_1
    del arg73_1
    del buf239
    buf243 = empty((512, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf242, (512, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 2), (1, 768), 0), alpha=1, beta=1, out=buf243)
    del arg160_1
    del arg161_1
    del buf242
    buf244 = reinterpret_tensor(buf241, (1, 512), (512, 1), 0); del buf241  # reuse
    buf245 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf246 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf240, (1, 512), (512, 1), 0); del buf240  # reuse
    buf248 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf249 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf250 = reinterpret_tensor(buf245, (), (), 0); del buf245  # reuse
    buf251 = buf250; del buf250  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_73(c_void_p(buf251.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del arg164_1
    del arg165_1
    return (buf251, buf244, buf247, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((50265, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg163_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg164_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    arg165_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
