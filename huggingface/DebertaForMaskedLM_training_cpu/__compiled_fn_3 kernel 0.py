
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


cpp_fused_add_bernoulli_embedding_mean_pow_sqrt_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr2 = in_out_ptr0;
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
                        tmp15.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_div_masked_fill_mul_rsub_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = in_ptr3[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 / tmp4;
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = static_cast<float>(0.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                    auto tmp12 = static_cast<float>(1.1111111111111112);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_2 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_5 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = static_cast<float>(768.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp9 - tmp13;
                        auto tmp15 = tmp14 * tmp14;
                        tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_9 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_10 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_17 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_24 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_25 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_32 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_39 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_40 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_47 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_50 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_54 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_55 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_57 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_62 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_65 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_69 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_70 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_77 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_sqrt_transpose_84 = async_compile.cpp('''
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
                        tmp5.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_85 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__softmax__to_copy_add_bitwise_not_detach_masked_fill_mul_rsub_transpose_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<bool>(0);
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = tmp4 ? tmp5 : tmp3;
                    auto tmp7 = tmp0 ? tmp5 : tmp6;
                    auto tmp8 = static_cast<float>(1.1111111111111112);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp9;
                    out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp6;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (192L*x0) + (2304L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                        tmp2.store(out_ptr4 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_87 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                auto tmp3 = c10::convert<bool>(tmp2);
                out_ptr0[static_cast<long>(x0)] = tmp3;
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
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = in_ptr3[static_cast<long>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 / tmp11;
                        auto tmp13 = tmp8 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(768.0);
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp0 - tmp4;
                        auto tmp6 = tmp5 * tmp5;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    tmp7.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = static_cast<float>(768.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-07);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp14 * tmp21;
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp26.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax__softmax_bitwise_not_detach_gelu_masked_fill_native_layer_norm_native_layer_norm_backward_nll_loss_forward_93 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (50265L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = std::log(tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    out_ptr2[static_cast<long>(x1 + (50265L*x0))] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 50265);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 50265L), "index out of bounds: 0 <= tmp8 < 50265L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (50265L*x0))];
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x0));
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
                    auto tmp0 = in_out_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr3[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr4[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr4[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr5[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr5[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr6[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr6[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr7[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr7[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr8[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr8[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr9[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr9[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr10[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr10[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr11[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr11[static_cast<long>(x1 + (512L*x0))] = tmp5;
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
                    auto tmp0 = in_out_ptr12[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr12[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (50265, 768), (768, 1))
    assert_size_stride(primals_76, (512, 768), (768, 1))
    assert_size_stride(primals_77, (2304, 768), (768, 1))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (3072, ), (1, ))
    assert_size_stride(primals_82, (768, 3072), (3072, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (2304, 768), (768, 1))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (3072, 768), (768, 1))
    assert_size_stride(primals_88, (3072, ), (1, ))
    assert_size_stride(primals_89, (768, 3072), (3072, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (2304, 768), (768, 1))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (3072, 768), (768, 1))
    assert_size_stride(primals_95, (3072, ), (1, ))
    assert_size_stride(primals_96, (768, 3072), (3072, 1))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (2304, 768), (768, 1))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (2304, 768), (768, 1))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768), (768, 1))
    assert_size_stride(primals_109, (3072, ), (1, ))
    assert_size_stride(primals_110, (768, 3072), (3072, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768), (768, 1))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (3072, 768), (768, 1))
    assert_size_stride(primals_116, (3072, ), (1, ))
    assert_size_stride(primals_117, (768, 3072), (3072, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (2304, 768), (768, 1))
    assert_size_stride(primals_120, (768, 768), (768, 1))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (2304, 768), (768, 1))
    assert_size_stride(primals_127, (768, 768), (768, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (3072, 768), (768, 1))
    assert_size_stride(primals_130, (3072, ), (1, ))
    assert_size_stride(primals_131, (768, 3072), (3072, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (2304, 768), (768, 1))
    assert_size_stride(primals_134, (768, 768), (768, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (3072, 768), (768, 1))
    assert_size_stride(primals_137, (3072, ), (1, ))
    assert_size_stride(primals_138, (768, 3072), (3072, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (2304, 768), (768, 1))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (2304, 768), (768, 1))
    assert_size_stride(primals_148, (768, 768), (768, 1))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (3072, 768), (768, 1))
    assert_size_stride(primals_151, (3072, ), (1, ))
    assert_size_stride(primals_152, (768, 3072), (3072, 1))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (2304, 768), (768, 1))
    assert_size_stride(primals_155, (768, 768), (768, 1))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (3072, 768), (768, 1))
    assert_size_stride(primals_158, (3072, ), (1, ))
    assert_size_stride(primals_159, (768, 3072), (3072, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (50265, 768), (768, 1))
    assert_size_stride(primals_166, (50265, ), (1, ))
    assert_size_stride(primals_167, (1, 512), (512, 1))
    assert_size_stride(primals_168, (1, 512), (512, 1))
    assert_size_stride(primals_169, (1, 512), (512, 1))
    buf0 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf2, (1, 512, 1), (512, 1, 1), 0); del buf2  # reuse
    buf4 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf38 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_bernoulli_embedding_mean_pow_sqrt_sub_0(c_void_p(buf3.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_75
    del primals_76
    aten.bernoulli_(buf5, 0.9)
    buf8 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf9 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_div_masked_fill_mul_rsub_1(c_void_p(buf5.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_2
    buf10 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (512, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 2304), (1, 768), 0), out=buf10)
    buf11 = reinterpret_tensor(buf5, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf5  # reuse
    buf515 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_2(c_void_p(buf10.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf515.data_ptr()))
    del primals_3
    buf12 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf10, (12, 64, 512), (192, 1, 2304), 64), out=buf12)
    buf13 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf12, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf12  # reuse
    buf15 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf16 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf54 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_3(c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf54.data_ptr()))
    aten.bernoulli_(buf17, 0.9)
    buf20 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf21 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf22 = buf11; del buf11  # reuse
    buf513 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_4(c_void_p(buf17.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf513.data_ptr()))
    del primals_4
    buf23 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf21, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf22, (12, 512, 64), (32768, 64, 1), 0), out=buf23)
    buf24 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
    cpp_fused_view_5(c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, buf24, reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
    del primals_79
    aten.bernoulli_(buf26, 0.9)
    buf29 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf30 = buf0; del buf0  # reuse
    buf31 = reinterpret_tensor(buf25, (1, 512, 768), (393216, 768, 1), 0); del buf25  # reuse
    buf32 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf33 = reinterpret_tensor(buf32, (1, 512, 1), (512, 1, 1), 0); del buf32  # reuse
    buf34 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_6(c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf34.data_ptr()))
    buf35 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, buf34, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf35)
    del primals_81
    buf36 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_7(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    buf37 = reinterpret_tensor(buf26, (512, 768), (768, 1), 0); del buf26  # reuse
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_83, buf36, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf37)
    del primals_83
    aten.bernoulli_(buf38, 0.9)
    buf41 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf42 = reinterpret_tensor(buf37, (1, 512, 768), (393216, 768, 1), 0); del buf37  # reuse
    buf43 = buf30; del buf30  # reuse
    buf44 = buf42; del buf42  # reuse
    buf45 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf45, (1, 512, 1), (512, 1, 1), 0); del buf45  # reuse
    buf47 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_8(c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_6
    buf48 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf47, reinterpret_tensor(primals_84, (768, 2304), (1, 768), 0), out=buf48)
    buf49 = reinterpret_tensor(buf38, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf38  # reuse
    buf512 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_9(c_void_p(buf48.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf512.data_ptr()))
    del primals_9
    buf50 = reinterpret_tensor(buf17, (12, 512, 512), (262144, 512, 1), 0); del buf17  # reuse
    # Source Nodes: [attention_scores_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf48, (12, 64, 512), (192, 1, 2304), 64), out=buf50)
    buf51 = buf13; del buf13  # reuse
    buf52 = reinterpret_tensor(buf50, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf50  # reuse
    buf53 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_10(c_void_p(buf52.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    aten.bernoulli_(buf54, 0.9)
    buf57 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf58 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf59 = buf49; del buf49  # reuse
    buf510 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_11(c_void_p(buf54.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf510.data_ptr()))
    del primals_10
    buf60 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf59, (12, 512, 64), (32768, 64, 1), 0), out=buf60)
    buf61 = reinterpret_tensor(buf59, (512, 768), (768, 1), 0); del buf59  # reuse
    cpp_fused_view_12(c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf60, (512, 768), (768, 1), 0); del buf60  # reuse
    # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf61, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf62)
    del primals_86
    buf63 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf76 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf101 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf114 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_13(c_void_p(buf4.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf114.data_ptr()))
    aten.bernoulli_(buf63, 0.9)
    buf66 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf67 = reinterpret_tensor(buf62, (1, 512, 768), (393216, 768, 1), 0); del buf62  # reuse
    buf68 = buf43; del buf43  # reuse
    buf69 = buf67; del buf67  # reuse
    buf70 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf71 = reinterpret_tensor(buf70, (1, 512, 1), (512, 1, 1), 0); del buf70  # reuse
    buf72 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14(c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_8
    buf73 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf72, reinterpret_tensor(primals_87, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del primals_88
    buf74 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_15(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf63, (512, 768), (768, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf74, reinterpret_tensor(primals_89, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf75)
    del primals_90
    aten.bernoulli_(buf76, 0.9)
    buf79 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf80 = reinterpret_tensor(buf75, (1, 512, 768), (393216, 768, 1), 0); del buf75  # reuse
    buf81 = buf68; del buf68  # reuse
    buf82 = buf80; del buf80  # reuse
    buf83 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf83, (1, 512, 1), (512, 1, 1), 0); del buf83  # reuse
    buf85 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_16(c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf85.data_ptr()))
    del primals_12
    buf86 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf85, reinterpret_tensor(primals_91, (768, 2304), (1, 768), 0), out=buf86)
    buf87 = reinterpret_tensor(buf76, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf76  # reuse
    buf509 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_17(c_void_p(buf86.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf509.data_ptr()))
    del primals_15
    buf88 = reinterpret_tensor(buf54, (12, 512, 512), (262144, 512, 1), 0); del buf54  # reuse
    # Source Nodes: [attention_scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf87, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf86, (12, 64, 512), (192, 1, 2304), 64), out=buf88)
    buf89 = buf51; del buf51  # reuse
    buf90 = reinterpret_tensor(buf88, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf88  # reuse
    buf91 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf92 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf130 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_18(c_void_p(buf90.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf130.data_ptr()))
    aten.bernoulli_(buf92, 0.9)
    buf95 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf96 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf97 = buf87; del buf87  # reuse
    buf507 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_19(c_void_p(buf92.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf507.data_ptr()))
    del primals_16
    buf98 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf96, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf97, (12, 512, 64), (32768, 64, 1), 0), out=buf98)
    buf99 = reinterpret_tensor(buf97, (512, 768), (768, 1), 0); del buf97  # reuse
    cpp_fused_view_20(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_93, buf99, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf100)
    del primals_93
    aten.bernoulli_(buf101, 0.9)
    buf104 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf105 = reinterpret_tensor(buf100, (1, 512, 768), (393216, 768, 1), 0); del buf100  # reuse
    buf106 = buf81; del buf81  # reuse
    buf107 = buf105; del buf105  # reuse
    buf108 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf108, (1, 512, 1), (512, 1, 1), 0); del buf108  # reuse
    buf110 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_21(c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf110.data_ptr()))
    del primals_14
    buf111 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_95, buf110, reinterpret_tensor(primals_94, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf111)
    del primals_95
    buf112 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    buf113 = reinterpret_tensor(buf101, (512, 768), (768, 1), 0); del buf101  # reuse
    # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_97, buf112, reinterpret_tensor(primals_96, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf113)
    del primals_97
    aten.bernoulli_(buf114, 0.9)
    buf117 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf118 = reinterpret_tensor(buf113, (1, 512, 768), (393216, 768, 1), 0); del buf113  # reuse
    buf119 = buf106; del buf106  # reuse
    buf120 = buf118; del buf118  # reuse
    buf121 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf122 = reinterpret_tensor(buf121, (1, 512, 1), (512, 1, 1), 0); del buf121  # reuse
    buf123 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_23(c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_18
    buf124 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf123, reinterpret_tensor(primals_98, (768, 2304), (1, 768), 0), out=buf124)
    buf125 = reinterpret_tensor(buf114, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf114  # reuse
    buf506 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_24(c_void_p(buf124.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf506.data_ptr()))
    del primals_21
    buf126 = reinterpret_tensor(buf92, (12, 512, 512), (262144, 512, 1), 0); del buf92  # reuse
    # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf125, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf124, (12, 64, 512), (192, 1, 2304), 64), out=buf126)
    buf127 = buf89; del buf89  # reuse
    buf128 = reinterpret_tensor(buf126, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf126  # reuse
    buf129 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_25(c_void_p(buf128.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    aten.bernoulli_(buf130, 0.9)
    buf133 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf134 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf135 = buf125; del buf125  # reuse
    buf504 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_26(c_void_p(buf130.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf504.data_ptr()))
    del primals_22
    buf136 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf134, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf135, (12, 512, 64), (32768, 64, 1), 0), out=buf136)
    buf137 = reinterpret_tensor(buf135, (512, 768), (768, 1), 0); del buf135  # reuse
    cpp_fused_view_27(c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    buf138 = reinterpret_tensor(buf136, (512, 768), (768, 1), 0); del buf136  # reuse
    # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf137, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf138)
    del primals_100
    buf139 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf152 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf177 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf190 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_28(c_void_p(buf4.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf190.data_ptr()))
    aten.bernoulli_(buf139, 0.9)
    buf142 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf143 = reinterpret_tensor(buf138, (1, 512, 768), (393216, 768, 1), 0); del buf138  # reuse
    buf144 = buf119; del buf119  # reuse
    buf145 = buf143; del buf143  # reuse
    buf146 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf147 = reinterpret_tensor(buf146, (1, 512, 1), (512, 1, 1), 0); del buf146  # reuse
    buf148 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_29(c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_20
    buf149 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf148, reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf149)
    del primals_102
    buf150 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = reinterpret_tensor(buf139, (512, 768), (768, 1), 0); del buf139  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf150, reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf151)
    del primals_104
    aten.bernoulli_(buf152, 0.9)
    buf155 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf156 = reinterpret_tensor(buf151, (1, 512, 768), (393216, 768, 1), 0); del buf151  # reuse
    buf157 = buf144; del buf144  # reuse
    buf158 = buf156; del buf156  # reuse
    buf159 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf160 = reinterpret_tensor(buf159, (1, 512, 1), (512, 1, 1), 0); del buf159  # reuse
    buf161 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_31(c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf161.data_ptr()))
    del primals_24
    buf162 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf161, reinterpret_tensor(primals_105, (768, 2304), (1, 768), 0), out=buf162)
    buf163 = reinterpret_tensor(buf152, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf152  # reuse
    buf503 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_32(c_void_p(buf162.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf503.data_ptr()))
    del primals_27
    buf164 = reinterpret_tensor(buf130, (12, 512, 512), (262144, 512, 1), 0); del buf130  # reuse
    # Source Nodes: [attention_scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf162, (12, 64, 512), (192, 1, 2304), 64), out=buf164)
    buf165 = buf127; del buf127  # reuse
    buf166 = reinterpret_tensor(buf164, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf164  # reuse
    buf167 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf168 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf206 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_33(c_void_p(buf166.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf206.data_ptr()))
    aten.bernoulli_(buf168, 0.9)
    buf171 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf172 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf173 = buf163; del buf163  # reuse
    buf501 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_34(c_void_p(buf168.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf501.data_ptr()))
    del primals_28
    buf174 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf173, (12, 512, 64), (32768, 64, 1), 0), out=buf174)
    buf175 = reinterpret_tensor(buf173, (512, 768), (768, 1), 0); del buf173  # reuse
    cpp_fused_view_35(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf174, (512, 768), (768, 1), 0); del buf174  # reuse
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_107, buf175, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf176)
    del primals_107
    aten.bernoulli_(buf177, 0.9)
    buf180 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf181 = reinterpret_tensor(buf176, (1, 512, 768), (393216, 768, 1), 0); del buf176  # reuse
    buf182 = buf157; del buf157  # reuse
    buf183 = buf181; del buf181  # reuse
    buf184 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf185 = reinterpret_tensor(buf184, (1, 512, 1), (512, 1, 1), 0); del buf184  # reuse
    buf186 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_36(c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf186.data_ptr()))
    del primals_26
    buf187 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_109, buf186, reinterpret_tensor(primals_108, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf187)
    del primals_109
    buf188 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_37(c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    buf189 = reinterpret_tensor(buf177, (512, 768), (768, 1), 0); del buf177  # reuse
    # Source Nodes: [hidden_states_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf188, reinterpret_tensor(primals_110, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf189)
    del primals_111
    aten.bernoulli_(buf190, 0.9)
    buf193 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf194 = reinterpret_tensor(buf189, (1, 512, 768), (393216, 768, 1), 0); del buf189  # reuse
    buf195 = buf182; del buf182  # reuse
    buf196 = buf194; del buf194  # reuse
    buf197 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf198 = reinterpret_tensor(buf197, (1, 512, 1), (512, 1, 1), 0); del buf197  # reuse
    buf199 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_38(c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_30
    buf200 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf199, reinterpret_tensor(primals_112, (768, 2304), (1, 768), 0), out=buf200)
    buf201 = reinterpret_tensor(buf190, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf190  # reuse
    buf500 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_39(c_void_p(buf200.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf500.data_ptr()))
    del primals_33
    buf202 = reinterpret_tensor(buf168, (12, 512, 512), (262144, 512, 1), 0); del buf168  # reuse
    # Source Nodes: [attention_scores_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf201, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf200, (12, 64, 512), (192, 1, 2304), 64), out=buf202)
    buf203 = buf165; del buf165  # reuse
    buf204 = reinterpret_tensor(buf202, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf202  # reuse
    buf205 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_40(c_void_p(buf204.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()))
    aten.bernoulli_(buf206, 0.9)
    buf209 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf210 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf211 = buf201; del buf201  # reuse
    buf498 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_41(c_void_p(buf206.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf498.data_ptr()))
    del primals_34
    buf212 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf210, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf211, (12, 512, 64), (32768, 64, 1), 0), out=buf212)
    buf213 = reinterpret_tensor(buf211, (512, 768), (768, 1), 0); del buf211  # reuse
    cpp_fused_view_42(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf212, (512, 768), (768, 1), 0); del buf212  # reuse
    # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf213, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf214)
    del primals_114
    buf215 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf228 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf253 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf266 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_43(c_void_p(buf4.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf266.data_ptr()))
    aten.bernoulli_(buf215, 0.9)
    buf218 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf219 = reinterpret_tensor(buf214, (1, 512, 768), (393216, 768, 1), 0); del buf214  # reuse
    buf220 = buf195; del buf195  # reuse
    buf221 = buf219; del buf219  # reuse
    buf222 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf223 = reinterpret_tensor(buf222, (1, 512, 1), (512, 1, 1), 0); del buf222  # reuse
    buf224 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_44(c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf224.data_ptr()))
    del primals_32
    buf225 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf224, reinterpret_tensor(primals_115, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf225)
    del primals_116
    buf226 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_45(c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = reinterpret_tensor(buf215, (512, 768), (768, 1), 0); del buf215  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf226, reinterpret_tensor(primals_117, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf227)
    del primals_118
    aten.bernoulli_(buf228, 0.9)
    buf231 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf232 = reinterpret_tensor(buf227, (1, 512, 768), (393216, 768, 1), 0); del buf227  # reuse
    buf233 = buf220; del buf220  # reuse
    buf234 = buf232; del buf232  # reuse
    buf235 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf236 = reinterpret_tensor(buf235, (1, 512, 1), (512, 1, 1), 0); del buf235  # reuse
    buf237 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_46(c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf237.data_ptr()))
    del primals_36
    buf238 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf237, reinterpret_tensor(primals_119, (768, 2304), (1, 768), 0), out=buf238)
    buf239 = reinterpret_tensor(buf228, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf228  # reuse
    buf497 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_47(c_void_p(buf238.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf497.data_ptr()))
    del primals_39
    buf240 = reinterpret_tensor(buf206, (12, 512, 512), (262144, 512, 1), 0); del buf206  # reuse
    # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf239, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf238, (12, 64, 512), (192, 1, 2304), 64), out=buf240)
    buf241 = buf203; del buf203  # reuse
    buf242 = reinterpret_tensor(buf240, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf240  # reuse
    buf243 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf244 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf282 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_48(c_void_p(buf242.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf282.data_ptr()))
    aten.bernoulli_(buf244, 0.9)
    buf247 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf248 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf249 = buf239; del buf239  # reuse
    buf495 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_49(c_void_p(buf244.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf495.data_ptr()))
    del primals_40
    buf250 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf248, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf249, (12, 512, 64), (32768, 64, 1), 0), out=buf250)
    buf251 = reinterpret_tensor(buf249, (512, 768), (768, 1), 0); del buf249  # reuse
    cpp_fused_view_50(c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = reinterpret_tensor(buf250, (512, 768), (768, 1), 0); del buf250  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf251, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf252)
    del primals_121
    aten.bernoulli_(buf253, 0.9)
    buf256 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf257 = reinterpret_tensor(buf252, (1, 512, 768), (393216, 768, 1), 0); del buf252  # reuse
    buf258 = buf233; del buf233  # reuse
    buf259 = buf257; del buf257  # reuse
    buf260 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf260, (1, 512, 1), (512, 1, 1), 0); del buf260  # reuse
    buf262 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_51(c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()))
    del primals_38
    buf263 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf262, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf263)
    del primals_123
    buf264 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_52(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf253, (512, 768), (768, 1), 0); del buf253  # reuse
    # Source Nodes: [hidden_states_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_125, buf264, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf265)
    del primals_125
    aten.bernoulli_(buf266, 0.9)
    buf269 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf270 = reinterpret_tensor(buf265, (1, 512, 768), (393216, 768, 1), 0); del buf265  # reuse
    buf271 = buf258; del buf258  # reuse
    buf272 = buf270; del buf270  # reuse
    buf273 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf274 = reinterpret_tensor(buf273, (1, 512, 1), (512, 1, 1), 0); del buf273  # reuse
    buf275 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_53(c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf275.data_ptr()))
    del primals_42
    buf276 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_126, (768, 2304), (1, 768), 0), out=buf276)
    buf277 = reinterpret_tensor(buf266, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf266  # reuse
    buf494 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_54(c_void_p(buf276.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf494.data_ptr()))
    del primals_45
    buf278 = reinterpret_tensor(buf244, (12, 512, 512), (262144, 512, 1), 0); del buf244  # reuse
    # Source Nodes: [attention_scores_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf277, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf276, (12, 64, 512), (192, 1, 2304), 64), out=buf278)
    buf279 = buf241; del buf241  # reuse
    buf280 = reinterpret_tensor(buf278, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf278  # reuse
    buf281 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_55(c_void_p(buf280.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    aten.bernoulli_(buf282, 0.9)
    buf285 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf286 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf287 = buf277; del buf277  # reuse
    buf492 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_56(c_void_p(buf282.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf492.data_ptr()))
    del primals_46
    buf288 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf286, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf287, (12, 512, 64), (32768, 64, 1), 0), out=buf288)
    buf289 = reinterpret_tensor(buf287, (512, 768), (768, 1), 0); del buf287  # reuse
    cpp_fused_view_57(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf288, (512, 768), (768, 1), 0); del buf288  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf289, reinterpret_tensor(primals_127, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf290)
    del primals_128
    buf291 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf304 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf329 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf342 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_58(c_void_p(buf4.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf342.data_ptr()))
    aten.bernoulli_(buf291, 0.9)
    buf294 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf295 = reinterpret_tensor(buf290, (1, 512, 768), (393216, 768, 1), 0); del buf290  # reuse
    buf296 = buf271; del buf271  # reuse
    buf297 = buf295; del buf295  # reuse
    buf298 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf298, (1, 512, 1), (512, 1, 1), 0); del buf298  # reuse
    buf300 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_59(c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_44
    buf301 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf300, reinterpret_tensor(primals_129, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf301)
    del primals_130
    buf302 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_60(c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf291, (512, 768), (768, 1), 0); del buf291  # reuse
    # Source Nodes: [hidden_states_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf302, reinterpret_tensor(primals_131, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf303)
    del primals_132
    aten.bernoulli_(buf304, 0.9)
    buf307 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf308 = reinterpret_tensor(buf303, (1, 512, 768), (393216, 768, 1), 0); del buf303  # reuse
    buf309 = buf296; del buf296  # reuse
    buf310 = buf308; del buf308  # reuse
    buf311 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf312 = reinterpret_tensor(buf311, (1, 512, 1), (512, 1, 1), 0); del buf311  # reuse
    buf313 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_61(c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_48
    buf314 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf313, reinterpret_tensor(primals_133, (768, 2304), (1, 768), 0), out=buf314)
    buf315 = reinterpret_tensor(buf304, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf304  # reuse
    buf491 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_62(c_void_p(buf314.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf491.data_ptr()))
    del primals_51
    buf316 = reinterpret_tensor(buf282, (12, 512, 512), (262144, 512, 1), 0); del buf282  # reuse
    # Source Nodes: [attention_scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf315, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf314, (12, 64, 512), (192, 1, 2304), 64), out=buf316)
    buf317 = buf279; del buf279  # reuse
    buf318 = reinterpret_tensor(buf316, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf316  # reuse
    buf319 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf320 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf358 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_63(c_void_p(buf318.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf358.data_ptr()))
    aten.bernoulli_(buf320, 0.9)
    buf323 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf324 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf325 = buf315; del buf315  # reuse
    buf489 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_64(c_void_p(buf320.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf489.data_ptr()))
    del primals_52
    buf326 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf324, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf325, (12, 512, 64), (32768, 64, 1), 0), out=buf326)
    buf327 = reinterpret_tensor(buf325, (512, 768), (768, 1), 0); del buf325  # reuse
    cpp_fused_view_65(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_135, buf327, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf328)
    del primals_135
    aten.bernoulli_(buf329, 0.9)
    buf332 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf333 = reinterpret_tensor(buf328, (1, 512, 768), (393216, 768, 1), 0); del buf328  # reuse
    buf334 = buf309; del buf309  # reuse
    buf335 = buf333; del buf333  # reuse
    buf336 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf337 = reinterpret_tensor(buf336, (1, 512, 1), (512, 1, 1), 0); del buf336  # reuse
    buf338 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_66(c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf338.data_ptr()))
    del primals_50
    buf339 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_137, buf338, reinterpret_tensor(primals_136, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf339)
    del primals_137
    buf340 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_67(c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = reinterpret_tensor(buf329, (512, 768), (768, 1), 0); del buf329  # reuse
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf340, reinterpret_tensor(primals_138, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf341)
    del primals_139
    aten.bernoulli_(buf342, 0.9)
    buf345 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf346 = reinterpret_tensor(buf341, (1, 512, 768), (393216, 768, 1), 0); del buf341  # reuse
    buf347 = buf334; del buf334  # reuse
    buf348 = buf346; del buf346  # reuse
    buf349 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf350 = reinterpret_tensor(buf349, (1, 512, 1), (512, 1, 1), 0); del buf349  # reuse
    buf351 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_68(c_void_p(buf348.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf351.data_ptr()))
    del primals_54
    buf352 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf351, reinterpret_tensor(primals_140, (768, 2304), (1, 768), 0), out=buf352)
    buf353 = reinterpret_tensor(buf342, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf342  # reuse
    buf488 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_69(c_void_p(buf352.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf488.data_ptr()))
    del primals_57
    buf354 = reinterpret_tensor(buf320, (12, 512, 512), (262144, 512, 1), 0); del buf320  # reuse
    # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf352, (12, 64, 512), (192, 1, 2304), 64), out=buf354)
    buf355 = buf317; del buf317  # reuse
    buf356 = reinterpret_tensor(buf354, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf354  # reuse
    buf357 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_70(c_void_p(buf356.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()))
    aten.bernoulli_(buf358, 0.9)
    buf361 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf362 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf363 = buf353; del buf353  # reuse
    buf486 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_71(c_void_p(buf358.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf486.data_ptr()))
    del primals_58
    buf364 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf362, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf363, (12, 512, 64), (32768, 64, 1), 0), out=buf364)
    buf365 = reinterpret_tensor(buf363, (512, 768), (768, 1), 0); del buf363  # reuse
    cpp_fused_view_72(c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    buf366 = reinterpret_tensor(buf364, (512, 768), (768, 1), 0); del buf364  # reuse
    # Source Nodes: [hidden_states_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf365, reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf366)
    del primals_142
    buf367 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf380 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf405 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf418 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_73(c_void_p(buf4.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf418.data_ptr()))
    aten.bernoulli_(buf367, 0.9)
    buf370 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf371 = reinterpret_tensor(buf366, (1, 512, 768), (393216, 768, 1), 0); del buf366  # reuse
    buf372 = buf347; del buf347  # reuse
    buf373 = buf371; del buf371  # reuse
    buf374 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf375 = reinterpret_tensor(buf374, (1, 512, 1), (512, 1, 1), 0); del buf374  # reuse
    buf376 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_74(c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf376.data_ptr()))
    del primals_56
    buf377 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf376, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf377)
    del primals_144
    buf378 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_75(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf367, (512, 768), (768, 1), 0); del buf367  # reuse
    # Source Nodes: [hidden_states_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf378, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf379)
    del primals_146
    aten.bernoulli_(buf380, 0.9)
    buf383 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf384 = reinterpret_tensor(buf379, (1, 512, 768), (393216, 768, 1), 0); del buf379  # reuse
    buf385 = buf372; del buf372  # reuse
    buf386 = buf384; del buf384  # reuse
    buf387 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf388 = reinterpret_tensor(buf387, (1, 512, 1), (512, 1, 1), 0); del buf387  # reuse
    buf389 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_76(c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf389.data_ptr()))
    del primals_60
    buf390 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf389, reinterpret_tensor(primals_147, (768, 2304), (1, 768), 0), out=buf390)
    buf391 = reinterpret_tensor(buf380, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf380  # reuse
    buf485 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_77(c_void_p(buf390.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf485.data_ptr()))
    del primals_63
    buf392 = reinterpret_tensor(buf358, (12, 512, 512), (262144, 512, 1), 0); del buf358  # reuse
    # Source Nodes: [attention_scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf391, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf390, (12, 64, 512), (192, 1, 2304), 64), out=buf392)
    buf393 = buf355; del buf355  # reuse
    buf394 = reinterpret_tensor(buf392, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf392  # reuse
    buf395 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf396 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf434 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_78(c_void_p(buf394.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf434.data_ptr()))
    aten.bernoulli_(buf396, 0.9)
    buf399 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf400 = buf16; del buf16  # reuse
    buf401 = buf391; del buf391  # reuse
    buf483 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_masked_fill_mul_rsub_transpose_79(c_void_p(buf396.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf483.data_ptr()))
    del primals_64
    buf402 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf400, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf401, (12, 512, 64), (32768, 64, 1), 0), out=buf402)
    buf403 = reinterpret_tensor(buf401, (512, 768), (768, 1), 0); del buf401  # reuse
    cpp_fused_view_80(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf402, (512, 768), (768, 1), 0); del buf402  # reuse
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf403, reinterpret_tensor(primals_148, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf404)
    del primals_149
    aten.bernoulli_(buf405, 0.9)
    buf408 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf409 = reinterpret_tensor(buf404, (1, 512, 768), (393216, 768, 1), 0); del buf404  # reuse
    buf410 = buf385; del buf385  # reuse
    buf411 = buf409; del buf409  # reuse
    buf412 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf412, (1, 512, 1), (512, 1, 1), 0); del buf412  # reuse
    buf414 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_81(c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf414.data_ptr()))
    del primals_62
    buf415 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf414, reinterpret_tensor(primals_150, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf415)
    del primals_151
    buf416 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_82(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf405, (512, 768), (768, 1), 0); del buf405  # reuse
    # Source Nodes: [hidden_states_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, buf416, reinterpret_tensor(primals_152, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf417)
    del primals_153
    aten.bernoulli_(buf418, 0.9)
    buf421 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf422 = reinterpret_tensor(buf417, (1, 512, 768), (393216, 768, 1), 0); del buf417  # reuse
    buf423 = buf410; del buf410  # reuse
    buf424 = buf422; del buf422  # reuse
    buf425 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf426 = reinterpret_tensor(buf425, (1, 512, 1), (512, 1, 1), 0); del buf425  # reuse
    buf427 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_83(c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_66
    buf428 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [qp_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf427, reinterpret_tensor(primals_154, (768, 2304), (1, 768), 0), out=buf428)
    buf429 = reinterpret_tensor(buf418, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf418  # reuse
    buf482 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_sqrt_transpose_84(c_void_p(buf428.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf482.data_ptr()))
    del primals_69
    buf430 = reinterpret_tensor(buf396, (12, 512, 512), (262144, 512, 1), 0); del buf396  # reuse
    # Source Nodes: [attention_scores_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf429, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf428, (12, 64, 512), (192, 1, 2304), 64), out=buf430)
    buf431 = buf393; del buf393  # reuse
    buf432 = reinterpret_tensor(buf430, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf430  # reuse
    buf433 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_85(c_void_p(buf432.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    del buf431
    aten.bernoulli_(buf434, 0.9)
    buf437 = empty((1, 12, 512, 512), device='cpu', dtype=torch.bool)
    buf438 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf481 = empty((1, 12, 512, 512), device='cpu', dtype=torch.float32)
    buf439 = buf429; del buf429  # reuse
    buf480 = empty_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_bitwise_not_detach_masked_fill_mul_rsub_transpose_86(c_void_p(buf434.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf480.data_ptr()))
    del buf432
    del buf433
    del buf434
    del primals_70
    buf440 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf438, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf439, (12, 512, 64), (32768, 64, 1), 0), out=buf440)
    buf441 = reinterpret_tensor(buf439, (512, 768), (768, 1), 0); del buf439  # reuse
    cpp_fused_view_87(c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = reinterpret_tensor(buf440, (512, 768), (768, 1), 0); del buf440  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf441, reinterpret_tensor(primals_155, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf442)
    del primals_156
    buf443 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf456 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_88(c_void_p(buf4.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf456.data_ptr()))
    aten.bernoulli_(buf443, 0.9)
    buf446 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf447 = reinterpret_tensor(buf442, (1, 512, 768), (393216, 768, 1), 0); del buf442  # reuse
    buf448 = buf423; del buf423  # reuse
    buf449 = buf447; del buf447  # reuse
    buf450 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf451 = reinterpret_tensor(buf450, (1, 512, 1), (512, 1, 1), 0); del buf450  # reuse
    buf452 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_89(c_void_p(buf449.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf452.data_ptr()))
    del primals_68
    buf453 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf452, reinterpret_tensor(primals_157, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf453)
    del primals_158
    buf454 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_90(c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    buf455 = reinterpret_tensor(buf443, (512, 768), (768, 1), 0); del buf443  # reuse
    # Source Nodes: [hidden_states_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf454, reinterpret_tensor(primals_159, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf455)
    del primals_160
    aten.bernoulli_(buf456, 0.9)
    buf459 = empty((1, 512, 768), device='cpu', dtype=torch.bool)
    buf460 = reinterpret_tensor(buf455, (1, 512, 768), (393216, 768, 1), 0); del buf455  # reuse
    buf461 = buf448; del buf448  # reuse
    buf462 = buf460; del buf460  # reuse
    buf463 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf464 = reinterpret_tensor(buf463, (1, 512, 1), (512, 1, 1), 0); del buf463  # reuse
    buf465 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_91(c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf465.data_ptr()))
    del primals_72
    del primals_74
    buf466 = reinterpret_tensor(buf456, (512, 768), (768, 1), 0); del buf456  # reuse
    # Source Nodes: [hidden_states_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf465, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf466)
    del primals_162
    buf467 = buf461; del buf461  # reuse
    buf468 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf470 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf471 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_92(c_void_p(buf466.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    del primals_164
    buf472 = empty((512, 50265), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf471, reinterpret_tensor(primals_165, (768, 50265), (1, 768), 0), alpha=1, beta=1, out=buf472)
    del primals_166
    buf473 = reinterpret_tensor(buf467, (512, 1), (1, 512), 0); del buf467  # reuse
    buf474 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf475 = empty((512, 50265), device='cpu', dtype=torch.float32)
    buf476 = empty((), device='cpu', dtype=torch.int64)
    buf478 = empty((), device='cpu', dtype=torch.float32)
    buf477 = empty((), device='cpu', dtype=torch.float32)
    buf516 = buf478; del buf478  # reuse
    buf479 = reinterpret_tensor(buf468, (1, 512, 1), (512, 1, 1), 0); del buf468  # reuse
    buf484 = buf394; del buf394  # reuse
    buf487 = buf356; del buf356  # reuse
    buf490 = buf318; del buf318  # reuse
    buf493 = buf280; del buf280  # reuse
    buf496 = buf242; del buf242  # reuse
    buf499 = buf204; del buf204  # reuse
    buf502 = buf166; del buf166  # reuse
    buf505 = buf128; del buf128  # reuse
    buf508 = buf90; del buf90  # reuse
    buf511 = buf52; del buf52  # reuse
    buf514 = buf14; del buf14  # reuse
    cpp_fused__log_softmax__softmax_bitwise_not_detach_gelu_masked_fill_native_layer_norm_native_layer_norm_backward_nll_loss_forward_93(c_void_p(buf516.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    return (buf516, reinterpret_tensor(buf472, (1, 512, 50265), (25735680, 50265, 1), 0), primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_163, primals_168, primals_169, primals_167, buf1, buf3, buf8, reinterpret_tensor(buf9, (512, 768), (768, 1), 0), buf20, buf24, buf29, buf31, buf33, buf34, buf35, buf36, buf41, buf44, buf46, buf47, buf57, buf61, buf66, buf69, buf71, buf72, buf73, buf74, buf79, buf82, buf84, buf85, buf95, buf99, buf104, buf107, buf109, buf110, buf111, buf112, buf117, buf120, buf122, buf123, buf133, buf137, buf142, buf145, buf147, buf148, buf149, buf150, buf155, buf158, buf160, buf161, buf171, buf175, buf180, buf183, buf185, buf186, buf187, buf188, buf193, buf196, buf198, buf199, buf209, buf213, buf218, buf221, buf223, buf224, buf225, buf226, buf231, buf234, buf236, buf237, buf247, buf251, buf256, buf259, buf261, buf262, buf263, buf264, buf269, buf272, buf274, buf275, buf285, buf289, buf294, buf297, buf299, buf300, buf301, buf302, buf307, buf310, buf312, buf313, buf323, buf327, buf332, buf335, buf337, buf338, buf339, buf340, buf345, buf348, buf350, buf351, buf361, buf365, buf370, buf373, buf375, buf376, buf377, buf378, buf383, buf386, buf388, buf389, buf399, buf403, buf408, buf411, buf413, buf414, buf415, buf416, buf421, buf424, buf426, buf427, buf437, buf441, buf446, buf449, buf451, buf452, buf453, buf454, buf459, buf462, buf464, buf465, buf466, buf470, buf471, buf475, buf477, reinterpret_tensor(primals_165, (50265, 768), (768, 1), 0), buf479, reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_157, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_155, (768, 768), (768, 1), 0), reinterpret_tensor(buf438, (12, 512, 512), (262144, 1, 512), 0), buf480, buf481, buf482, reinterpret_tensor(buf428, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_154, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_150, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_148, (768, 768), (768, 1), 0), reinterpret_tensor(buf400, (12, 512, 512), (262144, 1, 512), 0), buf483, buf484, buf485, reinterpret_tensor(buf390, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_147, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), reinterpret_tensor(buf362, (12, 512, 512), (262144, 1, 512), 0), buf486, buf487, buf488, reinterpret_tensor(buf352, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_140, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_136, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), reinterpret_tensor(buf324, (12, 512, 512), (262144, 1, 512), 0), buf489, buf490, buf491, reinterpret_tensor(buf314, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_133, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_129, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_127, (768, 768), (768, 1), 0), reinterpret_tensor(buf286, (12, 512, 512), (262144, 1, 512), 0), buf492, buf493, buf494, reinterpret_tensor(buf276, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_126, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(buf248, (12, 512, 512), (262144, 1, 512), 0), buf495, buf496, buf497, reinterpret_tensor(buf238, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_119, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (12, 512, 512), (262144, 1, 512), 0), buf498, buf499, buf500, reinterpret_tensor(buf200, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_112, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_110, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_108, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(buf172, (12, 512, 512), (262144, 1, 512), 0), buf501, buf502, buf503, reinterpret_tensor(buf162, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_105, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_103, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (12, 512, 512), (262144, 1, 512), 0), buf504, buf505, buf506, reinterpret_tensor(buf124, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_98, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_94, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(buf96, (12, 512, 512), (262144, 1, 512), 0), buf507, buf508, buf509, reinterpret_tensor(buf86, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_91, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_87, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), reinterpret_tensor(buf58, (12, 512, 512), (262144, 1, 512), 0), buf510, buf511, buf512, reinterpret_tensor(buf48, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_84, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_78, (768, 768), (768, 1), 0), reinterpret_tensor(buf21, (12, 512, 512), (262144, 1, 512), 0), buf513, buf514, buf515, reinterpret_tensor(buf10, (12, 512, 64), (192, 2304, 1), 64), reinterpret_tensor(primals_77, (2304, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((50265, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((50265, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((50265, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_168 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_169 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForMaskedLM', benchmark_compiled_module)
