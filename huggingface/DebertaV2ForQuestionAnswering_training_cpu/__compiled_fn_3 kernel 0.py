
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


cpp_fused_add_bernoulli_embedding_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128100);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128100L), "index out of bounds: 0 <= tmp3 < 128100L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 512);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp10);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_embedding_masked_fill_mul_native_layer_norm_rsub_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp11 = in_ptr5[static_cast<long>(x0)];
                    auto tmp14 = in_ptr6[static_cast<long>(x0)];
                    auto tmp22 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 128100);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 128100L), "index out of bounds: 0 <= tmp3 < 128100L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*tmp3)));
                    auto tmp6 = decltype(tmp5)(tmp5 + 512);
                    auto tmp7 = tmp5 < 0;
                    auto tmp8 = tmp7 ? tmp6 : tmp5;
                    TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*tmp8)));
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 - tmp12;
                    auto tmp15 = static_cast<float>(1536.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = static_cast<float>(1e-07);
                    auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                    auto tmp19 = 1 / std::sqrt(tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 * tmp20;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    auto tmp27 = static_cast<float>(0.0);
                    auto tmp28 = at::vec::Vectorized<float>(tmp27);
                    auto tmp29 = decltype(tmp28)::blendv(tmp26, tmp28, tmp22);
                    auto tmp30 = static_cast<float>(1.1111111111111112);
                    auto tmp31 = at::vec::Vectorized<float>(tmp30);
                    auto tmp32 = tmp29 * tmp31;
                    tmp21.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    tmp32.store(out_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp3)::blendv(tmp13, tmp3, tmp8);
                        auto tmp15 = tmp14 * tmp6;
                        auto tmp16 = tmp7 + tmp15;
                        tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp16);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_102 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_110 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_123 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_125 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_130 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_132 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_138 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_140 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_145 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_147 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_148 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_153 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_155 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_160 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_162 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_168 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_170 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_172 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_175 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax__to_copy_bitwise_not_clone_detach_masked_fill_mul_rsub_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_177 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_179 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        auto tmp13 = tmp7 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-07);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax__softmax_add_bitwise_not_clamp_clone_detach_div_embedding_masked_fill_native_layer_norm_native_layer_norm_backward_nll_loss_backward_nll_loss_forward_transpose_182 = async_compile.cpp('''
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
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       float* in_out_ptr32,
                       float* in_out_ptr33,
                       float* in_out_ptr34,
                       float* in_out_ptr35,
                       float* in_out_ptr36,
                       float* in_out_ptr37,
                       float* in_out_ptr38,
                       float* in_out_ptr39,
                       float* in_out_ptr40,
                       float* in_out_ptr41,
                       float* in_out_ptr42,
                       float* in_out_ptr43,
                       float* in_out_ptr44,
                       float* in_out_ptr45,
                       float* in_out_ptr46,
                       float* in_out_ptr47,
                       float* in_out_ptr48,
                       float* in_out_ptr49,
                       float* in_out_ptr50,
                       float* in_out_ptr51,
                       float* in_out_ptr52,
                       float* in_out_ptr53,
                       float* in_out_ptr54,
                       float* in_out_ptr55,
                       float* in_out_ptr56,
                       float* in_out_ptr57,
                       float* in_out_ptr58,
                       float* in_out_ptr59,
                       float* in_out_ptr60,
                       float* in_out_ptr61,
                       float* in_out_ptr62,
                       float* in_out_ptr63,
                       float* in_out_ptr64,
                       float* in_out_ptr65,
                       float* in_out_ptr66,
                       float* in_out_ptr67,
                       float* in_out_ptr68,
                       float* in_out_ptr69,
                       float* in_out_ptr70,
                       float* in_out_ptr71,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
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
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9,
                       float* out_ptr10,
                       bool* out_ptr11,
                       long* out_ptr12,
                       bool* out_ptr13,
                       long* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32,
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35,
                       float* out_ptr36,
                       float* out_ptr37,
                       float* out_ptr38)
{
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
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
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
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = out_ptr1[static_cast<long>(0L)];
            auto tmp4 = out_ptr4[static_cast<long>(0L)];
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 - tmp2;
            auto tmp5 = std::log(tmp4);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp3 - tmp6;
            tmp7.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                auto tmp1 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = out_ptr3[static_cast<long>(0L)];
            auto tmp4 = out_ptr6[static_cast<long>(0L)];
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 - tmp2;
            auto tmp5 = std::log(tmp4);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp3 - tmp6;
            tmp7.store(out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp6 = in_ptr2[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(512);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp7 = max_propagate_nan(tmp6, tmp1);
        auto tmp8 = min_propagate_nan(tmp7, tmp3);
        auto tmp9 = tmp8 != tmp3;
        auto tmp10 = tmp5 ? tmp4 : tmp1;
        auto tmp11 = decltype(tmp10)(tmp10 + 512);
        auto tmp12 = tmp10 < 0;
        auto tmp13 = tmp12 ? tmp11 : tmp10;
        TORCH_CHECK((0 <= tmp13) & (tmp13 < 512L), "index out of bounds: 0 <= tmp13 < 512L")
        auto tmp14 = out_ptr5[static_cast<long>(tmp13)];
        auto tmp15 = decltype(tmp14)(-tmp14);
        auto tmp16 = static_cast<float>(0.0);
        auto tmp17 = tmp5 ? tmp15 : tmp16;
        auto tmp18 = c10::convert<long>(tmp5);
        auto tmp19 = c10::convert<float>(tmp18);
        auto tmp20 = tmp17 / tmp19;
        auto tmp21 = tmp9 ? tmp8 : tmp1;
        auto tmp22 = decltype(tmp21)(tmp21 + 512);
        auto tmp23 = tmp21 < 0;
        auto tmp24 = tmp23 ? tmp22 : tmp21;
        TORCH_CHECK((0 <= tmp24) & (tmp24 < 512L), "index out of bounds: 0 <= tmp24 < 512L")
        auto tmp25 = out_ptr7[static_cast<long>(tmp24)];
        auto tmp26 = decltype(tmp25)(-tmp25);
        auto tmp27 = tmp9 ? tmp26 : tmp16;
        auto tmp28 = c10::convert<long>(tmp9);
        auto tmp29 = c10::convert<float>(tmp28);
        auto tmp30 = tmp27 / tmp29;
        auto tmp31 = decltype(tmp20)(tmp20 + tmp30);
        auto tmp32 = static_cast<float>(2.0);
        auto tmp33 = tmp31 / tmp32;
        out_ptr8[static_cast<long>(0L)] = tmp5;
        out_ptr9[static_cast<long>(0L)] = tmp9;
        out_ptr10[static_cast<long>(0L)] = tmp33;
    }
    {
        auto tmp0 = in_ptr2[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(512);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        out_ptr11[static_cast<long>(0L)] = tmp5;
        out_ptr12[static_cast<long>(0L)] = tmp6;
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(512);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        out_ptr13[static_cast<long>(0L)] = tmp5;
        out_ptr14[static_cast<long>(0L)] = tmp6;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1536.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-07);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1536.0);
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr15 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr16 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr7[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr17 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr10[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr18 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr11 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr13[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr13[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr19 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr16[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr16[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr20 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr19[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr19[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr21 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr22[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr22[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr22 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr25[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr18[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr25[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr23 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr28[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr20[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr28[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr24 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr31[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr22[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr31[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr25 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr34[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr24[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr34[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr26 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr37[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr26[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr37[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr27 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr40[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr28[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr40[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr28 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr42 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr43[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr30[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr43[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr29 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr46[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr32[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr46[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr30 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr47 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr48 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr49[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr34[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr49[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr31 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr51 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr52[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr36[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr52[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr32 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr53 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr54 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr55[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr38[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr55[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr33 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr56 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr57 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr58[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr40[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr58[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr34 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr61[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr42[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr61[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr35 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr62 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr63 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr64[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr44[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr64[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr36 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr66 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr67[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr46[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr67[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr37 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr68 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr70[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr48[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<bool>(0);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    in_out_ptr70[static_cast<long>(x1 + (512L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x2 + (64L*x0) + (1536L*x1)));
                        tmp0.store(out_ptr38 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr71 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394 = args
    args.clear()
    assert_size_stride(primals_1, (128100, 1536), (1536, 1))
    assert_size_stride(primals_2, (512, 1536), (1536, 1))
    assert_size_stride(primals_3, (1536, ), (1, ))
    assert_size_stride(primals_4, (1536, ), (1, ))
    assert_size_stride(primals_5, (1536, 1536), (1536, 1))
    assert_size_stride(primals_6, (1536, ), (1, ))
    assert_size_stride(primals_7, (1536, 1536), (1536, 1))
    assert_size_stride(primals_8, (1536, ), (1, ))
    assert_size_stride(primals_9, (1536, 1536), (1536, 1))
    assert_size_stride(primals_10, (1536, ), (1, ))
    assert_size_stride(primals_11, (1536, 1536), (1536, 1))
    assert_size_stride(primals_12, (1536, ), (1, ))
    assert_size_stride(primals_13, (1536, ), (1, ))
    assert_size_stride(primals_14, (1536, ), (1, ))
    assert_size_stride(primals_15, (6144, 1536), (1536, 1))
    assert_size_stride(primals_16, (6144, ), (1, ))
    assert_size_stride(primals_17, (1536, 6144), (6144, 1))
    assert_size_stride(primals_18, (1536, ), (1, ))
    assert_size_stride(primals_19, (1536, ), (1, ))
    assert_size_stride(primals_20, (1536, ), (1, ))
    assert_size_stride(primals_21, (1536, 1536), (1536, 1))
    assert_size_stride(primals_22, (1536, ), (1, ))
    assert_size_stride(primals_23, (1536, 1536), (1536, 1))
    assert_size_stride(primals_24, (1536, ), (1, ))
    assert_size_stride(primals_25, (1536, 1536), (1536, 1))
    assert_size_stride(primals_26, (1536, ), (1, ))
    assert_size_stride(primals_27, (1536, 1536), (1536, 1))
    assert_size_stride(primals_28, (1536, ), (1, ))
    assert_size_stride(primals_29, (1536, ), (1, ))
    assert_size_stride(primals_30, (1536, ), (1, ))
    assert_size_stride(primals_31, (6144, 1536), (1536, 1))
    assert_size_stride(primals_32, (6144, ), (1, ))
    assert_size_stride(primals_33, (1536, 6144), (6144, 1))
    assert_size_stride(primals_34, (1536, ), (1, ))
    assert_size_stride(primals_35, (1536, ), (1, ))
    assert_size_stride(primals_36, (1536, ), (1, ))
    assert_size_stride(primals_37, (1536, 1536), (1536, 1))
    assert_size_stride(primals_38, (1536, ), (1, ))
    assert_size_stride(primals_39, (1536, 1536), (1536, 1))
    assert_size_stride(primals_40, (1536, ), (1, ))
    assert_size_stride(primals_41, (1536, 1536), (1536, 1))
    assert_size_stride(primals_42, (1536, ), (1, ))
    assert_size_stride(primals_43, (1536, 1536), (1536, 1))
    assert_size_stride(primals_44, (1536, ), (1, ))
    assert_size_stride(primals_45, (1536, ), (1, ))
    assert_size_stride(primals_46, (1536, ), (1, ))
    assert_size_stride(primals_47, (6144, 1536), (1536, 1))
    assert_size_stride(primals_48, (6144, ), (1, ))
    assert_size_stride(primals_49, (1536, 6144), (6144, 1))
    assert_size_stride(primals_50, (1536, ), (1, ))
    assert_size_stride(primals_51, (1536, ), (1, ))
    assert_size_stride(primals_52, (1536, ), (1, ))
    assert_size_stride(primals_53, (1536, 1536), (1536, 1))
    assert_size_stride(primals_54, (1536, ), (1, ))
    assert_size_stride(primals_55, (1536, 1536), (1536, 1))
    assert_size_stride(primals_56, (1536, ), (1, ))
    assert_size_stride(primals_57, (1536, 1536), (1536, 1))
    assert_size_stride(primals_58, (1536, ), (1, ))
    assert_size_stride(primals_59, (1536, 1536), (1536, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (1536, ), (1, ))
    assert_size_stride(primals_62, (1536, ), (1, ))
    assert_size_stride(primals_63, (6144, 1536), (1536, 1))
    assert_size_stride(primals_64, (6144, ), (1, ))
    assert_size_stride(primals_65, (1536, 6144), (6144, 1))
    assert_size_stride(primals_66, (1536, ), (1, ))
    assert_size_stride(primals_67, (1536, ), (1, ))
    assert_size_stride(primals_68, (1536, ), (1, ))
    assert_size_stride(primals_69, (1536, 1536), (1536, 1))
    assert_size_stride(primals_70, (1536, ), (1, ))
    assert_size_stride(primals_71, (1536, 1536), (1536, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (1536, 1536), (1536, 1))
    assert_size_stride(primals_74, (1536, ), (1, ))
    assert_size_stride(primals_75, (1536, 1536), (1536, 1))
    assert_size_stride(primals_76, (1536, ), (1, ))
    assert_size_stride(primals_77, (1536, ), (1, ))
    assert_size_stride(primals_78, (1536, ), (1, ))
    assert_size_stride(primals_79, (6144, 1536), (1536, 1))
    assert_size_stride(primals_80, (6144, ), (1, ))
    assert_size_stride(primals_81, (1536, 6144), (6144, 1))
    assert_size_stride(primals_82, (1536, ), (1, ))
    assert_size_stride(primals_83, (1536, ), (1, ))
    assert_size_stride(primals_84, (1536, ), (1, ))
    assert_size_stride(primals_85, (1536, 1536), (1536, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (1536, 1536), (1536, 1))
    assert_size_stride(primals_88, (1536, ), (1, ))
    assert_size_stride(primals_89, (1536, 1536), (1536, 1))
    assert_size_stride(primals_90, (1536, ), (1, ))
    assert_size_stride(primals_91, (1536, 1536), (1536, 1))
    assert_size_stride(primals_92, (1536, ), (1, ))
    assert_size_stride(primals_93, (1536, ), (1, ))
    assert_size_stride(primals_94, (1536, ), (1, ))
    assert_size_stride(primals_95, (6144, 1536), (1536, 1))
    assert_size_stride(primals_96, (6144, ), (1, ))
    assert_size_stride(primals_97, (1536, 6144), (6144, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (1536, ), (1, ))
    assert_size_stride(primals_100, (1536, ), (1, ))
    assert_size_stride(primals_101, (1536, 1536), (1536, 1))
    assert_size_stride(primals_102, (1536, ), (1, ))
    assert_size_stride(primals_103, (1536, 1536), (1536, 1))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_105, (1536, 1536), (1536, 1))
    assert_size_stride(primals_106, (1536, ), (1, ))
    assert_size_stride(primals_107, (1536, 1536), (1536, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (1536, ), (1, ))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (6144, 1536), (1536, 1))
    assert_size_stride(primals_112, (6144, ), (1, ))
    assert_size_stride(primals_113, (1536, 6144), (6144, 1))
    assert_size_stride(primals_114, (1536, ), (1, ))
    assert_size_stride(primals_115, (1536, ), (1, ))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (1536, 1536), (1536, 1))
    assert_size_stride(primals_118, (1536, ), (1, ))
    assert_size_stride(primals_119, (1536, 1536), (1536, 1))
    assert_size_stride(primals_120, (1536, ), (1, ))
    assert_size_stride(primals_121, (1536, 1536), (1536, 1))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_123, (1536, 1536), (1536, 1))
    assert_size_stride(primals_124, (1536, ), (1, ))
    assert_size_stride(primals_125, (1536, ), (1, ))
    assert_size_stride(primals_126, (1536, ), (1, ))
    assert_size_stride(primals_127, (6144, 1536), (1536, 1))
    assert_size_stride(primals_128, (6144, ), (1, ))
    assert_size_stride(primals_129, (1536, 6144), (6144, 1))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_131, (1536, ), (1, ))
    assert_size_stride(primals_132, (1536, ), (1, ))
    assert_size_stride(primals_133, (1536, 1536), (1536, 1))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_135, (1536, 1536), (1536, 1))
    assert_size_stride(primals_136, (1536, ), (1, ))
    assert_size_stride(primals_137, (1536, 1536), (1536, 1))
    assert_size_stride(primals_138, (1536, ), (1, ))
    assert_size_stride(primals_139, (1536, 1536), (1536, 1))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_141, (1536, ), (1, ))
    assert_size_stride(primals_142, (1536, ), (1, ))
    assert_size_stride(primals_143, (6144, 1536), (1536, 1))
    assert_size_stride(primals_144, (6144, ), (1, ))
    assert_size_stride(primals_145, (1536, 6144), (6144, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (1536, ), (1, ))
    assert_size_stride(primals_148, (1536, ), (1, ))
    assert_size_stride(primals_149, (1536, 1536), (1536, 1))
    assert_size_stride(primals_150, (1536, ), (1, ))
    assert_size_stride(primals_151, (1536, 1536), (1536, 1))
    assert_size_stride(primals_152, (1536, ), (1, ))
    assert_size_stride(primals_153, (1536, 1536), (1536, 1))
    assert_size_stride(primals_154, (1536, ), (1, ))
    assert_size_stride(primals_155, (1536, 1536), (1536, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (1536, ), (1, ))
    assert_size_stride(primals_158, (1536, ), (1, ))
    assert_size_stride(primals_159, (6144, 1536), (1536, 1))
    assert_size_stride(primals_160, (6144, ), (1, ))
    assert_size_stride(primals_161, (1536, 6144), (6144, 1))
    assert_size_stride(primals_162, (1536, ), (1, ))
    assert_size_stride(primals_163, (1536, ), (1, ))
    assert_size_stride(primals_164, (1536, ), (1, ))
    assert_size_stride(primals_165, (1536, 1536), (1536, 1))
    assert_size_stride(primals_166, (1536, ), (1, ))
    assert_size_stride(primals_167, (1536, 1536), (1536, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (1536, 1536), (1536, 1))
    assert_size_stride(primals_170, (1536, ), (1, ))
    assert_size_stride(primals_171, (1536, 1536), (1536, 1))
    assert_size_stride(primals_172, (1536, ), (1, ))
    assert_size_stride(primals_173, (1536, ), (1, ))
    assert_size_stride(primals_174, (1536, ), (1, ))
    assert_size_stride(primals_175, (6144, 1536), (1536, 1))
    assert_size_stride(primals_176, (6144, ), (1, ))
    assert_size_stride(primals_177, (1536, 6144), (6144, 1))
    assert_size_stride(primals_178, (1536, ), (1, ))
    assert_size_stride(primals_179, (1536, ), (1, ))
    assert_size_stride(primals_180, (1536, ), (1, ))
    assert_size_stride(primals_181, (1536, 1536), (1536, 1))
    assert_size_stride(primals_182, (1536, ), (1, ))
    assert_size_stride(primals_183, (1536, 1536), (1536, 1))
    assert_size_stride(primals_184, (1536, ), (1, ))
    assert_size_stride(primals_185, (1536, 1536), (1536, 1))
    assert_size_stride(primals_186, (1536, ), (1, ))
    assert_size_stride(primals_187, (1536, 1536), (1536, 1))
    assert_size_stride(primals_188, (1536, ), (1, ))
    assert_size_stride(primals_189, (1536, ), (1, ))
    assert_size_stride(primals_190, (1536, ), (1, ))
    assert_size_stride(primals_191, (6144, 1536), (1536, 1))
    assert_size_stride(primals_192, (6144, ), (1, ))
    assert_size_stride(primals_193, (1536, 6144), (6144, 1))
    assert_size_stride(primals_194, (1536, ), (1, ))
    assert_size_stride(primals_195, (1536, ), (1, ))
    assert_size_stride(primals_196, (1536, ), (1, ))
    assert_size_stride(primals_197, (1536, 1536), (1536, 1))
    assert_size_stride(primals_198, (1536, ), (1, ))
    assert_size_stride(primals_199, (1536, 1536), (1536, 1))
    assert_size_stride(primals_200, (1536, ), (1, ))
    assert_size_stride(primals_201, (1536, 1536), (1536, 1))
    assert_size_stride(primals_202, (1536, ), (1, ))
    assert_size_stride(primals_203, (1536, 1536), (1536, 1))
    assert_size_stride(primals_204, (1536, ), (1, ))
    assert_size_stride(primals_205, (1536, ), (1, ))
    assert_size_stride(primals_206, (1536, ), (1, ))
    assert_size_stride(primals_207, (6144, 1536), (1536, 1))
    assert_size_stride(primals_208, (6144, ), (1, ))
    assert_size_stride(primals_209, (1536, 6144), (6144, 1))
    assert_size_stride(primals_210, (1536, ), (1, ))
    assert_size_stride(primals_211, (1536, ), (1, ))
    assert_size_stride(primals_212, (1536, ), (1, ))
    assert_size_stride(primals_213, (1536, 1536), (1536, 1))
    assert_size_stride(primals_214, (1536, ), (1, ))
    assert_size_stride(primals_215, (1536, 1536), (1536, 1))
    assert_size_stride(primals_216, (1536, ), (1, ))
    assert_size_stride(primals_217, (1536, 1536), (1536, 1))
    assert_size_stride(primals_218, (1536, ), (1, ))
    assert_size_stride(primals_219, (1536, 1536), (1536, 1))
    assert_size_stride(primals_220, (1536, ), (1, ))
    assert_size_stride(primals_221, (1536, ), (1, ))
    assert_size_stride(primals_222, (1536, ), (1, ))
    assert_size_stride(primals_223, (6144, 1536), (1536, 1))
    assert_size_stride(primals_224, (6144, ), (1, ))
    assert_size_stride(primals_225, (1536, 6144), (6144, 1))
    assert_size_stride(primals_226, (1536, ), (1, ))
    assert_size_stride(primals_227, (1536, ), (1, ))
    assert_size_stride(primals_228, (1536, ), (1, ))
    assert_size_stride(primals_229, (1536, 1536), (1536, 1))
    assert_size_stride(primals_230, (1536, ), (1, ))
    assert_size_stride(primals_231, (1536, 1536), (1536, 1))
    assert_size_stride(primals_232, (1536, ), (1, ))
    assert_size_stride(primals_233, (1536, 1536), (1536, 1))
    assert_size_stride(primals_234, (1536, ), (1, ))
    assert_size_stride(primals_235, (1536, 1536), (1536, 1))
    assert_size_stride(primals_236, (1536, ), (1, ))
    assert_size_stride(primals_237, (1536, ), (1, ))
    assert_size_stride(primals_238, (1536, ), (1, ))
    assert_size_stride(primals_239, (6144, 1536), (1536, 1))
    assert_size_stride(primals_240, (6144, ), (1, ))
    assert_size_stride(primals_241, (1536, 6144), (6144, 1))
    assert_size_stride(primals_242, (1536, ), (1, ))
    assert_size_stride(primals_243, (1536, ), (1, ))
    assert_size_stride(primals_244, (1536, ), (1, ))
    assert_size_stride(primals_245, (1536, 1536), (1536, 1))
    assert_size_stride(primals_246, (1536, ), (1, ))
    assert_size_stride(primals_247, (1536, 1536), (1536, 1))
    assert_size_stride(primals_248, (1536, ), (1, ))
    assert_size_stride(primals_249, (1536, 1536), (1536, 1))
    assert_size_stride(primals_250, (1536, ), (1, ))
    assert_size_stride(primals_251, (1536, 1536), (1536, 1))
    assert_size_stride(primals_252, (1536, ), (1, ))
    assert_size_stride(primals_253, (1536, ), (1, ))
    assert_size_stride(primals_254, (1536, ), (1, ))
    assert_size_stride(primals_255, (6144, 1536), (1536, 1))
    assert_size_stride(primals_256, (6144, ), (1, ))
    assert_size_stride(primals_257, (1536, 6144), (6144, 1))
    assert_size_stride(primals_258, (1536, ), (1, ))
    assert_size_stride(primals_259, (1536, ), (1, ))
    assert_size_stride(primals_260, (1536, ), (1, ))
    assert_size_stride(primals_261, (1536, 1536), (1536, 1))
    assert_size_stride(primals_262, (1536, ), (1, ))
    assert_size_stride(primals_263, (1536, 1536), (1536, 1))
    assert_size_stride(primals_264, (1536, ), (1, ))
    assert_size_stride(primals_265, (1536, 1536), (1536, 1))
    assert_size_stride(primals_266, (1536, ), (1, ))
    assert_size_stride(primals_267, (1536, 1536), (1536, 1))
    assert_size_stride(primals_268, (1536, ), (1, ))
    assert_size_stride(primals_269, (1536, ), (1, ))
    assert_size_stride(primals_270, (1536, ), (1, ))
    assert_size_stride(primals_271, (6144, 1536), (1536, 1))
    assert_size_stride(primals_272, (6144, ), (1, ))
    assert_size_stride(primals_273, (1536, 6144), (6144, 1))
    assert_size_stride(primals_274, (1536, ), (1, ))
    assert_size_stride(primals_275, (1536, ), (1, ))
    assert_size_stride(primals_276, (1536, ), (1, ))
    assert_size_stride(primals_277, (1536, 1536), (1536, 1))
    assert_size_stride(primals_278, (1536, ), (1, ))
    assert_size_stride(primals_279, (1536, 1536), (1536, 1))
    assert_size_stride(primals_280, (1536, ), (1, ))
    assert_size_stride(primals_281, (1536, 1536), (1536, 1))
    assert_size_stride(primals_282, (1536, ), (1, ))
    assert_size_stride(primals_283, (1536, 1536), (1536, 1))
    assert_size_stride(primals_284, (1536, ), (1, ))
    assert_size_stride(primals_285, (1536, ), (1, ))
    assert_size_stride(primals_286, (1536, ), (1, ))
    assert_size_stride(primals_287, (6144, 1536), (1536, 1))
    assert_size_stride(primals_288, (6144, ), (1, ))
    assert_size_stride(primals_289, (1536, 6144), (6144, 1))
    assert_size_stride(primals_290, (1536, ), (1, ))
    assert_size_stride(primals_291, (1536, ), (1, ))
    assert_size_stride(primals_292, (1536, ), (1, ))
    assert_size_stride(primals_293, (1536, 1536), (1536, 1))
    assert_size_stride(primals_294, (1536, ), (1, ))
    assert_size_stride(primals_295, (1536, 1536), (1536, 1))
    assert_size_stride(primals_296, (1536, ), (1, ))
    assert_size_stride(primals_297, (1536, 1536), (1536, 1))
    assert_size_stride(primals_298, (1536, ), (1, ))
    assert_size_stride(primals_299, (1536, 1536), (1536, 1))
    assert_size_stride(primals_300, (1536, ), (1, ))
    assert_size_stride(primals_301, (1536, ), (1, ))
    assert_size_stride(primals_302, (1536, ), (1, ))
    assert_size_stride(primals_303, (6144, 1536), (1536, 1))
    assert_size_stride(primals_304, (6144, ), (1, ))
    assert_size_stride(primals_305, (1536, 6144), (6144, 1))
    assert_size_stride(primals_306, (1536, ), (1, ))
    assert_size_stride(primals_307, (1536, ), (1, ))
    assert_size_stride(primals_308, (1536, ), (1, ))
    assert_size_stride(primals_309, (1536, 1536), (1536, 1))
    assert_size_stride(primals_310, (1536, ), (1, ))
    assert_size_stride(primals_311, (1536, 1536), (1536, 1))
    assert_size_stride(primals_312, (1536, ), (1, ))
    assert_size_stride(primals_313, (1536, 1536), (1536, 1))
    assert_size_stride(primals_314, (1536, ), (1, ))
    assert_size_stride(primals_315, (1536, 1536), (1536, 1))
    assert_size_stride(primals_316, (1536, ), (1, ))
    assert_size_stride(primals_317, (1536, ), (1, ))
    assert_size_stride(primals_318, (1536, ), (1, ))
    assert_size_stride(primals_319, (6144, 1536), (1536, 1))
    assert_size_stride(primals_320, (6144, ), (1, ))
    assert_size_stride(primals_321, (1536, 6144), (6144, 1))
    assert_size_stride(primals_322, (1536, ), (1, ))
    assert_size_stride(primals_323, (1536, ), (1, ))
    assert_size_stride(primals_324, (1536, ), (1, ))
    assert_size_stride(primals_325, (1536, 1536), (1536, 1))
    assert_size_stride(primals_326, (1536, ), (1, ))
    assert_size_stride(primals_327, (1536, 1536), (1536, 1))
    assert_size_stride(primals_328, (1536, ), (1, ))
    assert_size_stride(primals_329, (1536, 1536), (1536, 1))
    assert_size_stride(primals_330, (1536, ), (1, ))
    assert_size_stride(primals_331, (1536, 1536), (1536, 1))
    assert_size_stride(primals_332, (1536, ), (1, ))
    assert_size_stride(primals_333, (1536, ), (1, ))
    assert_size_stride(primals_334, (1536, ), (1, ))
    assert_size_stride(primals_335, (6144, 1536), (1536, 1))
    assert_size_stride(primals_336, (6144, ), (1, ))
    assert_size_stride(primals_337, (1536, 6144), (6144, 1))
    assert_size_stride(primals_338, (1536, ), (1, ))
    assert_size_stride(primals_339, (1536, ), (1, ))
    assert_size_stride(primals_340, (1536, ), (1, ))
    assert_size_stride(primals_341, (1536, 1536), (1536, 1))
    assert_size_stride(primals_342, (1536, ), (1, ))
    assert_size_stride(primals_343, (1536, 1536), (1536, 1))
    assert_size_stride(primals_344, (1536, ), (1, ))
    assert_size_stride(primals_345, (1536, 1536), (1536, 1))
    assert_size_stride(primals_346, (1536, ), (1, ))
    assert_size_stride(primals_347, (1536, 1536), (1536, 1))
    assert_size_stride(primals_348, (1536, ), (1, ))
    assert_size_stride(primals_349, (1536, ), (1, ))
    assert_size_stride(primals_350, (1536, ), (1, ))
    assert_size_stride(primals_351, (6144, 1536), (1536, 1))
    assert_size_stride(primals_352, (6144, ), (1, ))
    assert_size_stride(primals_353, (1536, 6144), (6144, 1))
    assert_size_stride(primals_354, (1536, ), (1, ))
    assert_size_stride(primals_355, (1536, ), (1, ))
    assert_size_stride(primals_356, (1536, ), (1, ))
    assert_size_stride(primals_357, (1536, 1536), (1536, 1))
    assert_size_stride(primals_358, (1536, ), (1, ))
    assert_size_stride(primals_359, (1536, 1536), (1536, 1))
    assert_size_stride(primals_360, (1536, ), (1, ))
    assert_size_stride(primals_361, (1536, 1536), (1536, 1))
    assert_size_stride(primals_362, (1536, ), (1, ))
    assert_size_stride(primals_363, (1536, 1536), (1536, 1))
    assert_size_stride(primals_364, (1536, ), (1, ))
    assert_size_stride(primals_365, (1536, ), (1, ))
    assert_size_stride(primals_366, (1536, ), (1, ))
    assert_size_stride(primals_367, (6144, 1536), (1536, 1))
    assert_size_stride(primals_368, (6144, ), (1, ))
    assert_size_stride(primals_369, (1536, 6144), (6144, 1))
    assert_size_stride(primals_370, (1536, ), (1, ))
    assert_size_stride(primals_371, (1536, ), (1, ))
    assert_size_stride(primals_372, (1536, ), (1, ))
    assert_size_stride(primals_373, (1536, 1536), (1536, 1))
    assert_size_stride(primals_374, (1536, ), (1, ))
    assert_size_stride(primals_375, (1536, 1536), (1536, 1))
    assert_size_stride(primals_376, (1536, ), (1, ))
    assert_size_stride(primals_377, (1536, 1536), (1536, 1))
    assert_size_stride(primals_378, (1536, ), (1, ))
    assert_size_stride(primals_379, (1536, 1536), (1536, 1))
    assert_size_stride(primals_380, (1536, ), (1, ))
    assert_size_stride(primals_381, (1536, ), (1, ))
    assert_size_stride(primals_382, (1536, ), (1, ))
    assert_size_stride(primals_383, (6144, 1536), (1536, 1))
    assert_size_stride(primals_384, (6144, ), (1, ))
    assert_size_stride(primals_385, (1536, 6144), (6144, 1))
    assert_size_stride(primals_386, (1536, ), (1, ))
    assert_size_stride(primals_387, (1536, ), (1, ))
    assert_size_stride(primals_388, (1536, ), (1, ))
    assert_size_stride(primals_389, (2, 1536), (1536, 1))
    assert_size_stride(primals_390, (2, ), (1, ))
    assert_size_stride(primals_391, (1, 512), (512, 1))
    assert_size_stride(primals_392, (1, 512), (512, 1))
    assert_size_stride(primals_393, (1, ), (1, ))
    assert_size_stride(primals_394, (1, ), (1, ))
    buf0 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf29 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf42 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_bernoulli_embedding_native_layer_norm_0(c_void_p(primals_392.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf42.data_ptr()))
    aten.bernoulli_(buf5, 0.9)
    buf8 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf3 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf9 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_embedding_masked_fill_mul_native_layer_norm_rsub_view_1(c_void_p(buf5.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_1
    del primals_2
    buf10 = reinterpret_tensor(buf5, (512, 1536), (1536, 1), 0); del buf5  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, buf9, reinterpret_tensor(primals_5, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf10)
    del primals_6
    buf11 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf9, reinterpret_tensor(primals_7, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf11)
    del primals_8
    buf12 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_0_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, buf9, reinterpret_tensor(primals_9, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf12)
    del primals_10
    buf13 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf11, (24, 64, 512), (64, 1, 1536), 0); del buf11  # reuse
    cpp_fused_clone_div_sqrt_2(c_void_p(buf14.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf13.data_ptr()))
    buf15 = empty((24, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attention_scores, scale, truediv], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf13, (24, 512, 64), (32768, 64, 1), 0), buf14, out=buf15)
    buf16 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf17 = reinterpret_tensor(buf15, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf15  # reuse
    buf18 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf19 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf20 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf61 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_3(c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf61.data_ptr()))
    aten.bernoulli_(buf20, 0.9)
    buf23 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf24 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf25 = reinterpret_tensor(buf10, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf10  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_4(c_void_p(buf20.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf12, (24, 512, 64), (32768, 64, 1), 0); del buf12  # reuse
    # Source Nodes: [context_layer], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf25, (24, 512, 64), (32768, 64, 1), 0), out=buf26)
    buf27 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_5(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = reinterpret_tensor(buf26, (512, 1536), (1536, 1), 0); del buf26  # reuse
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf27, reinterpret_tensor(primals_11, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf28)
    del primals_12
    aten.bernoulli_(buf29, 0.9)
    buf32 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf33 = reinterpret_tensor(buf28, (1, 512, 1536), (786432, 1536, 1), 0); del buf28  # reuse
    buf34 = buf0; del buf0  # reuse
    buf35 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf37 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf38 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_6(c_void_p(buf33.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_4
    buf39 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, buf38, reinterpret_tensor(primals_15, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf39)
    del primals_16
    buf40 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_7(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf33, (512, 1536), (1536, 1), 0); del buf33  # reuse
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf40, reinterpret_tensor(primals_17, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf41)
    del primals_18
    aten.bernoulli_(buf42, 0.9)
    buf45 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf46 = reinterpret_tensor(buf41, (1, 512, 1536), (786432, 1536, 1), 0); del buf41  # reuse
    buf47 = buf34; del buf34  # reuse
    buf48 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf50 = buf29; del buf29  # reuse
    buf51 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_8(c_void_p(buf46.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_14
    buf52 = reinterpret_tensor(buf46, (512, 1536), (1536, 1), 0); del buf46  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf51, reinterpret_tensor(primals_21, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf52)
    del primals_22
    buf53 = reinterpret_tensor(buf42, (512, 1536), (1536, 1), 0); del buf42  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf51, reinterpret_tensor(primals_23, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf53)
    del primals_24
    buf54 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_1_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf51, reinterpret_tensor(primals_25, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf54)
    del primals_26
    buf55 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf56 = reinterpret_tensor(buf53, (24, 64, 512), (64, 1, 1536), 0); del buf53  # reuse
    cpp_fused_clone_div_sqrt_9(c_void_p(buf56.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf55.data_ptr()))
    buf57 = reinterpret_tensor(buf20, (24, 512, 512), (262144, 512, 1), 0); del buf20  # reuse
    # Source Nodes: [attention_scores_3, scale, truediv_1], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf55, (24, 512, 64), (32768, 64, 1), 0), buf56, out=buf57)
    buf58 = buf16; del buf16  # reuse
    buf59 = reinterpret_tensor(buf57, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf57  # reuse
    buf60 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_10(c_void_p(buf59.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()))
    aten.bernoulli_(buf61, 0.9)
    buf64 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf65 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf66 = reinterpret_tensor(buf52, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf52  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_11(c_void_p(buf61.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf54, (24, 512, 64), (32768, 64, 1), 0); del buf54  # reuse
    # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf66, (24, 512, 64), (32768, 64, 1), 0), out=buf67)
    buf68 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_12(c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    buf69 = reinterpret_tensor(buf67, (512, 1536), (1536, 1), 0); del buf67  # reuse
    # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf68, reinterpret_tensor(primals_27, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf69)
    del primals_28
    buf70 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf83 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf111 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf124 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_13(c_void_p(buf4.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf124.data_ptr()))
    aten.bernoulli_(buf70, 0.9)
    buf73 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf74 = reinterpret_tensor(buf69, (1, 512, 1536), (786432, 1536, 1), 0); del buf69  # reuse
    buf75 = buf47; del buf47  # reuse
    buf76 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf78 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf79 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_14(c_void_p(buf74.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_20
    buf80 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf79, reinterpret_tensor(primals_31, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf80)
    del primals_32
    buf81 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_15(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    buf82 = reinterpret_tensor(buf74, (512, 1536), (1536, 1), 0); del buf74  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_34, buf81, reinterpret_tensor(primals_33, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf82)
    del primals_34
    aten.bernoulli_(buf83, 0.9)
    buf86 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf87 = reinterpret_tensor(buf82, (1, 512, 1536), (786432, 1536, 1), 0); del buf82  # reuse
    buf88 = buf75; del buf75  # reuse
    buf89 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf91 = buf70; del buf70  # reuse
    buf92 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_16(c_void_p(buf87.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_30
    buf93 = reinterpret_tensor(buf87, (512, 1536), (1536, 1), 0); del buf87  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf92, reinterpret_tensor(primals_37, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf93)
    del primals_38
    buf94 = reinterpret_tensor(buf83, (512, 1536), (1536, 1), 0); del buf83  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, buf92, reinterpret_tensor(primals_39, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf94)
    del primals_40
    buf95 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_2_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, buf92, reinterpret_tensor(primals_41, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf95)
    del primals_42
    buf96 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf97 = reinterpret_tensor(buf94, (24, 64, 512), (64, 1, 1536), 0); del buf94  # reuse
    cpp_fused_clone_div_sqrt_17(c_void_p(buf97.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf96.data_ptr()))
    buf98 = reinterpret_tensor(buf61, (24, 512, 512), (262144, 512, 1), 0); del buf61  # reuse
    # Source Nodes: [attention_scores_6, scale, truediv_2], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf96, (24, 512, 64), (32768, 64, 1), 0), buf97, out=buf98)
    buf99 = buf58; del buf58  # reuse
    buf100 = reinterpret_tensor(buf98, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf98  # reuse
    buf101 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf102 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf143 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_18(c_void_p(buf100.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf143.data_ptr()))
    aten.bernoulli_(buf102, 0.9)
    buf105 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf106 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf107 = reinterpret_tensor(buf93, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf93  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_19(c_void_p(buf102.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    buf108 = reinterpret_tensor(buf95, (24, 512, 64), (32768, 64, 1), 0); del buf95  # reuse
    # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf106, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf107, (24, 512, 64), (32768, 64, 1), 0), out=buf108)
    buf109 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_20(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf108, (512, 1536), (1536, 1), 0); del buf108  # reuse
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf109, reinterpret_tensor(primals_43, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf110)
    del primals_44
    aten.bernoulli_(buf111, 0.9)
    buf114 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf115 = reinterpret_tensor(buf110, (1, 512, 1536), (786432, 1536, 1), 0); del buf110  # reuse
    buf116 = buf88; del buf88  # reuse
    buf117 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf119 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf120 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_21(c_void_p(buf115.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    del primals_36
    buf121 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf120, reinterpret_tensor(primals_47, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf121)
    del primals_48
    buf122 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf115, (512, 1536), (1536, 1), 0); del buf115  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf122, reinterpret_tensor(primals_49, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf123)
    del primals_50
    aten.bernoulli_(buf124, 0.9)
    buf127 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf128 = reinterpret_tensor(buf123, (1, 512, 1536), (786432, 1536, 1), 0); del buf123  # reuse
    buf129 = buf116; del buf116  # reuse
    buf130 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf132 = buf111; del buf111  # reuse
    buf133 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_23(c_void_p(buf128.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del primals_46
    buf134 = reinterpret_tensor(buf128, (512, 1536), (1536, 1), 0); del buf128  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf133, reinterpret_tensor(primals_53, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf134)
    del primals_54
    buf135 = reinterpret_tensor(buf124, (512, 1536), (1536, 1), 0); del buf124  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf133, reinterpret_tensor(primals_55, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf135)
    del primals_56
    buf136 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_3_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, buf133, reinterpret_tensor(primals_57, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf136)
    del primals_58
    buf137 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf135, (24, 64, 512), (64, 1, 1536), 0); del buf135  # reuse
    cpp_fused_clone_div_sqrt_24(c_void_p(buf138.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf137.data_ptr()))
    buf139 = reinterpret_tensor(buf102, (24, 512, 512), (262144, 512, 1), 0); del buf102  # reuse
    # Source Nodes: [attention_scores_9, scale, truediv_3], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf137, (24, 512, 64), (32768, 64, 1), 0), buf138, out=buf139)
    buf140 = buf99; del buf99  # reuse
    buf141 = reinterpret_tensor(buf139, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf139  # reuse
    buf142 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_25(c_void_p(buf141.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    aten.bernoulli_(buf143, 0.9)
    buf146 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf147 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf134, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf134  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_26(c_void_p(buf143.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf136, (24, 512, 64), (32768, 64, 1), 0); del buf136  # reuse
    # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf147, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf148, (24, 512, 64), (32768, 64, 1), 0), out=buf149)
    buf150 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_27(c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = reinterpret_tensor(buf149, (512, 1536), (1536, 1), 0); del buf149  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf150, reinterpret_tensor(primals_59, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf151)
    del primals_60
    buf152 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf165 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf193 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf206 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_28(c_void_p(buf4.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf206.data_ptr()))
    aten.bernoulli_(buf152, 0.9)
    buf155 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf156 = reinterpret_tensor(buf151, (1, 512, 1536), (786432, 1536, 1), 0); del buf151  # reuse
    buf157 = buf129; del buf129  # reuse
    buf158 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf160 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf161 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_29(c_void_p(buf156.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del primals_52
    buf162 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_64, buf161, reinterpret_tensor(primals_63, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf162)
    del primals_64
    buf163 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf156, (512, 1536), (1536, 1), 0); del buf156  # reuse
    # Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf163, reinterpret_tensor(primals_65, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf164)
    del primals_66
    aten.bernoulli_(buf165, 0.9)
    buf168 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf169 = reinterpret_tensor(buf164, (1, 512, 1536), (786432, 1536, 1), 0); del buf164  # reuse
    buf170 = buf157; del buf157  # reuse
    buf171 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf173 = buf152; del buf152  # reuse
    buf174 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_31(c_void_p(buf169.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_62
    buf175 = reinterpret_tensor(buf169, (512, 1536), (1536, 1), 0); del buf169  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, buf174, reinterpret_tensor(primals_69, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf175)
    del primals_70
    buf176 = reinterpret_tensor(buf165, (512, 1536), (1536, 1), 0); del buf165  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf174, reinterpret_tensor(primals_71, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf176)
    del primals_72
    buf177 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_4_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf174, reinterpret_tensor(primals_73, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf177)
    del primals_74
    buf178 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf176, (24, 64, 512), (64, 1, 1536), 0); del buf176  # reuse
    cpp_fused_clone_div_sqrt_32(c_void_p(buf179.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf178.data_ptr()))
    buf180 = reinterpret_tensor(buf143, (24, 512, 512), (262144, 512, 1), 0); del buf143  # reuse
    # Source Nodes: [attention_scores_12, scale, truediv_4], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf178, (24, 512, 64), (32768, 64, 1), 0), buf179, out=buf180)
    buf181 = buf140; del buf140  # reuse
    buf182 = reinterpret_tensor(buf180, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf180  # reuse
    buf183 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf184 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf225 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_33(c_void_p(buf182.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf225.data_ptr()))
    aten.bernoulli_(buf184, 0.9)
    buf187 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf188 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf189 = reinterpret_tensor(buf175, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf175  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_34(c_void_p(buf184.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = reinterpret_tensor(buf177, (24, 512, 64), (32768, 64, 1), 0); del buf177  # reuse
    # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf189, (24, 512, 64), (32768, 64, 1), 0), out=buf190)
    buf191 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_35(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = reinterpret_tensor(buf190, (512, 1536), (1536, 1), 0); del buf190  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, buf191, reinterpret_tensor(primals_75, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf192)
    del primals_76
    aten.bernoulli_(buf193, 0.9)
    buf196 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf197 = reinterpret_tensor(buf192, (1, 512, 1536), (786432, 1536, 1), 0); del buf192  # reuse
    buf198 = buf170; del buf170  # reuse
    buf199 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf201 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf202 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_36(c_void_p(buf197.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del primals_68
    buf203 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf202, reinterpret_tensor(primals_79, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf203)
    del primals_80
    buf204 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_37(c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf197, (512, 1536), (1536, 1), 0); del buf197  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf204, reinterpret_tensor(primals_81, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf205)
    del primals_82
    aten.bernoulli_(buf206, 0.9)
    buf209 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf210 = reinterpret_tensor(buf205, (1, 512, 1536), (786432, 1536, 1), 0); del buf205  # reuse
    buf211 = buf198; del buf198  # reuse
    buf212 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf214 = buf193; del buf193  # reuse
    buf215 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_38(c_void_p(buf210.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del primals_78
    buf216 = reinterpret_tensor(buf210, (512, 1536), (1536, 1), 0); del buf210  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf215, reinterpret_tensor(primals_85, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf216)
    del primals_86
    buf217 = reinterpret_tensor(buf206, (512, 1536), (1536, 1), 0); del buf206  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf215, reinterpret_tensor(primals_87, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf217)
    del primals_88
    buf218 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_5_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf215, reinterpret_tensor(primals_89, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf218)
    del primals_90
    buf219 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf217, (24, 64, 512), (64, 1, 1536), 0); del buf217  # reuse
    cpp_fused_clone_div_sqrt_39(c_void_p(buf220.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf219.data_ptr()))
    buf221 = reinterpret_tensor(buf184, (24, 512, 512), (262144, 512, 1), 0); del buf184  # reuse
    # Source Nodes: [attention_scores_15, scale, truediv_5], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf219, (24, 512, 64), (32768, 64, 1), 0), buf220, out=buf221)
    buf222 = buf181; del buf181  # reuse
    buf223 = reinterpret_tensor(buf221, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf221  # reuse
    buf224 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_40(c_void_p(buf223.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    aten.bernoulli_(buf225, 0.9)
    buf228 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf229 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf230 = reinterpret_tensor(buf216, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf216  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_41(c_void_p(buf225.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf218, (24, 512, 64), (32768, 64, 1), 0); del buf218  # reuse
    # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf230, (24, 512, 64), (32768, 64, 1), 0), out=buf231)
    buf232 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_42(c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf231, (512, 1536), (1536, 1), 0); del buf231  # reuse
    # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf232, reinterpret_tensor(primals_91, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf233)
    del primals_92
    buf234 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf247 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf275 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf288 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_43(c_void_p(buf4.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf288.data_ptr()))
    aten.bernoulli_(buf234, 0.9)
    buf237 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf238 = reinterpret_tensor(buf233, (1, 512, 1536), (786432, 1536, 1), 0); del buf233  # reuse
    buf239 = buf211; del buf211  # reuse
    buf240 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf242 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf243 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_44(c_void_p(buf238.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    del primals_84
    buf244 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf243, reinterpret_tensor(primals_95, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf244)
    del primals_96
    buf245 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_45(c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf238, (512, 1536), (1536, 1), 0); del buf238  # reuse
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf245, reinterpret_tensor(primals_97, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf246)
    del primals_98
    aten.bernoulli_(buf247, 0.9)
    buf250 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf251 = reinterpret_tensor(buf246, (1, 512, 1536), (786432, 1536, 1), 0); del buf246  # reuse
    buf252 = buf239; del buf239  # reuse
    buf253 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf255 = buf234; del buf234  # reuse
    buf256 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_46(c_void_p(buf251.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del primals_94
    buf257 = reinterpret_tensor(buf251, (512, 1536), (1536, 1), 0); del buf251  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf256, reinterpret_tensor(primals_101, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf257)
    del primals_102
    buf258 = reinterpret_tensor(buf247, (512, 1536), (1536, 1), 0); del buf247  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf256, reinterpret_tensor(primals_103, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf258)
    del primals_104
    buf259 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_6_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf256, reinterpret_tensor(primals_105, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf259)
    del primals_106
    buf260 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf258, (24, 64, 512), (64, 1, 1536), 0); del buf258  # reuse
    cpp_fused_clone_div_sqrt_47(c_void_p(buf261.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf260.data_ptr()))
    buf262 = reinterpret_tensor(buf225, (24, 512, 512), (262144, 512, 1), 0); del buf225  # reuse
    # Source Nodes: [attention_scores_18, scale, truediv_6], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf260, (24, 512, 64), (32768, 64, 1), 0), buf261, out=buf262)
    buf263 = buf222; del buf222  # reuse
    buf264 = reinterpret_tensor(buf262, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf262  # reuse
    buf265 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf266 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf307 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_48(c_void_p(buf264.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf307.data_ptr()))
    aten.bernoulli_(buf266, 0.9)
    buf269 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf270 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf271 = reinterpret_tensor(buf257, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf257  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_49(c_void_p(buf266.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    buf272 = reinterpret_tensor(buf259, (24, 512, 64), (32768, 64, 1), 0); del buf259  # reuse
    # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf271, (24, 512, 64), (32768, 64, 1), 0), out=buf272)
    buf273 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_50(c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf272, (512, 1536), (1536, 1), 0); del buf272  # reuse
    # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf273, reinterpret_tensor(primals_107, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf274)
    del primals_108
    aten.bernoulli_(buf275, 0.9)
    buf278 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf279 = reinterpret_tensor(buf274, (1, 512, 1536), (786432, 1536, 1), 0); del buf274  # reuse
    buf280 = buf252; del buf252  # reuse
    buf281 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf283 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf284 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_51(c_void_p(buf279.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    del primals_100
    buf285 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf284, reinterpret_tensor(primals_111, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf285)
    del primals_112
    buf286 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_52(c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf279, (512, 1536), (1536, 1), 0); del buf279  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf286, reinterpret_tensor(primals_113, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf287)
    del primals_114
    aten.bernoulli_(buf288, 0.9)
    buf291 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf292 = reinterpret_tensor(buf287, (1, 512, 1536), (786432, 1536, 1), 0); del buf287  # reuse
    buf293 = buf280; del buf280  # reuse
    buf294 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf296 = buf275; del buf275  # reuse
    buf297 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_53(c_void_p(buf292.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del primals_110
    buf298 = reinterpret_tensor(buf292, (512, 1536), (1536, 1), 0); del buf292  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf297, reinterpret_tensor(primals_117, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf298)
    del primals_118
    buf299 = reinterpret_tensor(buf288, (512, 1536), (1536, 1), 0); del buf288  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf297, reinterpret_tensor(primals_119, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf299)
    del primals_120
    buf300 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_7_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf297, reinterpret_tensor(primals_121, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf300)
    del primals_122
    buf301 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf299, (24, 64, 512), (64, 1, 1536), 0); del buf299  # reuse
    cpp_fused_clone_div_sqrt_54(c_void_p(buf302.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf301.data_ptr()))
    buf303 = reinterpret_tensor(buf266, (24, 512, 512), (262144, 512, 1), 0); del buf266  # reuse
    # Source Nodes: [attention_scores_21, scale, truediv_7], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf301, (24, 512, 64), (32768, 64, 1), 0), buf302, out=buf303)
    buf304 = buf263; del buf263  # reuse
    buf305 = reinterpret_tensor(buf303, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf303  # reuse
    buf306 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_55(c_void_p(buf305.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    aten.bernoulli_(buf307, 0.9)
    buf310 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf311 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf312 = reinterpret_tensor(buf298, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf298  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_56(c_void_p(buf307.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    buf313 = reinterpret_tensor(buf300, (24, 512, 64), (32768, 64, 1), 0); del buf300  # reuse
    # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf311, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf312, (24, 512, 64), (32768, 64, 1), 0), out=buf313)
    buf314 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_57(c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = reinterpret_tensor(buf313, (512, 1536), (1536, 1), 0); del buf313  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf314, reinterpret_tensor(primals_123, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf315)
    del primals_124
    buf316 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf329 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf357 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf370 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_58(c_void_p(buf4.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf370.data_ptr()))
    aten.bernoulli_(buf316, 0.9)
    buf319 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf320 = reinterpret_tensor(buf315, (1, 512, 1536), (786432, 1536, 1), 0); del buf315  # reuse
    buf321 = buf293; del buf293  # reuse
    buf322 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf324 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf325 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_59(c_void_p(buf320.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_116
    buf326 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf325, reinterpret_tensor(primals_127, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf326)
    del primals_128
    buf327 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_60(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = reinterpret_tensor(buf320, (512, 1536), (1536, 1), 0); del buf320  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf327, reinterpret_tensor(primals_129, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf328)
    del primals_130
    aten.bernoulli_(buf329, 0.9)
    buf332 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf333 = reinterpret_tensor(buf328, (1, 512, 1536), (786432, 1536, 1), 0); del buf328  # reuse
    buf334 = buf321; del buf321  # reuse
    buf335 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf337 = buf316; del buf316  # reuse
    buf338 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_61(c_void_p(buf333.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del primals_126
    buf339 = reinterpret_tensor(buf333, (512, 1536), (1536, 1), 0); del buf333  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf338, reinterpret_tensor(primals_133, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf339)
    del primals_134
    buf340 = reinterpret_tensor(buf329, (512, 1536), (1536, 1), 0); del buf329  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf338, reinterpret_tensor(primals_135, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf340)
    del primals_136
    buf341 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_8_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf338, reinterpret_tensor(primals_137, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf341)
    del primals_138
    buf342 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf343 = reinterpret_tensor(buf340, (24, 64, 512), (64, 1, 1536), 0); del buf340  # reuse
    cpp_fused_clone_div_sqrt_62(c_void_p(buf343.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf342.data_ptr()))
    buf344 = reinterpret_tensor(buf307, (24, 512, 512), (262144, 512, 1), 0); del buf307  # reuse
    # Source Nodes: [attention_scores_24, scale, truediv_8], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf342, (24, 512, 64), (32768, 64, 1), 0), buf343, out=buf344)
    buf345 = buf304; del buf304  # reuse
    buf346 = reinterpret_tensor(buf344, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf344  # reuse
    buf347 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf348 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf389 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_63(c_void_p(buf346.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf389.data_ptr()))
    aten.bernoulli_(buf348, 0.9)
    buf351 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf352 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf353 = reinterpret_tensor(buf339, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf339  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_64(c_void_p(buf348.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    buf354 = reinterpret_tensor(buf341, (24, 512, 64), (32768, 64, 1), 0); del buf341  # reuse
    # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf352, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf353, (24, 512, 64), (32768, 64, 1), 0), out=buf354)
    buf355 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_65(c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    buf356 = reinterpret_tensor(buf354, (512, 1536), (1536, 1), 0); del buf354  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf355, reinterpret_tensor(primals_139, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf356)
    del primals_140
    aten.bernoulli_(buf357, 0.9)
    buf360 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf361 = reinterpret_tensor(buf356, (1, 512, 1536), (786432, 1536, 1), 0); del buf356  # reuse
    buf362 = buf334; del buf334  # reuse
    buf363 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf365 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf366 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_66(c_void_p(buf361.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    del primals_132
    buf367 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf366, reinterpret_tensor(primals_143, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf367)
    del primals_144
    buf368 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_67(c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = reinterpret_tensor(buf361, (512, 1536), (1536, 1), 0); del buf361  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf368, reinterpret_tensor(primals_145, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf369)
    del primals_146
    aten.bernoulli_(buf370, 0.9)
    buf373 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf374 = reinterpret_tensor(buf369, (1, 512, 1536), (786432, 1536, 1), 0); del buf369  # reuse
    buf375 = buf362; del buf362  # reuse
    buf376 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf378 = buf357; del buf357  # reuse
    buf379 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_68(c_void_p(buf374.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del primals_142
    buf380 = reinterpret_tensor(buf374, (512, 1536), (1536, 1), 0); del buf374  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf379, reinterpret_tensor(primals_149, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf380)
    del primals_150
    buf381 = reinterpret_tensor(buf370, (512, 1536), (1536, 1), 0); del buf370  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf379, reinterpret_tensor(primals_151, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf381)
    del primals_152
    buf382 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_9_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, buf379, reinterpret_tensor(primals_153, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf382)
    del primals_154
    buf383 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf384 = reinterpret_tensor(buf381, (24, 64, 512), (64, 1, 1536), 0); del buf381  # reuse
    cpp_fused_clone_div_sqrt_69(c_void_p(buf384.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf383.data_ptr()))
    buf385 = reinterpret_tensor(buf348, (24, 512, 512), (262144, 512, 1), 0); del buf348  # reuse
    # Source Nodes: [attention_scores_27, scale, truediv_9], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf383, (24, 512, 64), (32768, 64, 1), 0), buf384, out=buf385)
    buf386 = buf345; del buf345  # reuse
    buf387 = reinterpret_tensor(buf385, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf385  # reuse
    buf388 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_70(c_void_p(buf387.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()))
    aten.bernoulli_(buf389, 0.9)
    buf392 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf393 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf394 = reinterpret_tensor(buf380, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf380  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_71(c_void_p(buf389.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = reinterpret_tensor(buf382, (24, 512, 64), (32768, 64, 1), 0); del buf382  # reuse
    # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf393, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf394, (24, 512, 64), (32768, 64, 1), 0), out=buf395)
    buf396 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_72(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = reinterpret_tensor(buf395, (512, 1536), (1536, 1), 0); del buf395  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf396, reinterpret_tensor(primals_155, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf397)
    del primals_156
    buf398 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf411 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf439 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf452 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_73(c_void_p(buf4.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf452.data_ptr()))
    aten.bernoulli_(buf398, 0.9)
    buf401 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf402 = reinterpret_tensor(buf397, (1, 512, 1536), (786432, 1536, 1), 0); del buf397  # reuse
    buf403 = buf375; del buf375  # reuse
    buf404 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf406 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf407 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_74(c_void_p(buf402.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    del primals_148
    buf408 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf407, reinterpret_tensor(primals_159, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf408)
    del primals_160
    buf409 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_75(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf402, (512, 1536), (1536, 1), 0); del buf402  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf409, reinterpret_tensor(primals_161, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf410)
    del primals_162
    aten.bernoulli_(buf411, 0.9)
    buf414 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf415 = reinterpret_tensor(buf410, (1, 512, 1536), (786432, 1536, 1), 0); del buf410  # reuse
    buf416 = buf403; del buf403  # reuse
    buf417 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf419 = buf398; del buf398  # reuse
    buf420 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_76(c_void_p(buf415.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()))
    del primals_158
    buf421 = reinterpret_tensor(buf415, (512, 1536), (1536, 1), 0); del buf415  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf420, reinterpret_tensor(primals_165, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf421)
    del primals_166
    buf422 = reinterpret_tensor(buf411, (512, 1536), (1536, 1), 0); del buf411  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, buf420, reinterpret_tensor(primals_167, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf422)
    del primals_168
    buf423 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_10_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, buf420, reinterpret_tensor(primals_169, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf423)
    del primals_170
    buf424 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf425 = reinterpret_tensor(buf422, (24, 64, 512), (64, 1, 1536), 0); del buf422  # reuse
    cpp_fused_clone_div_sqrt_77(c_void_p(buf425.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf424.data_ptr()))
    buf426 = reinterpret_tensor(buf389, (24, 512, 512), (262144, 512, 1), 0); del buf389  # reuse
    # Source Nodes: [attention_scores_30, scale, truediv_10], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf424, (24, 512, 64), (32768, 64, 1), 0), buf425, out=buf426)
    buf427 = buf386; del buf386  # reuse
    buf428 = reinterpret_tensor(buf426, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf426  # reuse
    buf429 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf430 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf471 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_78(c_void_p(buf428.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf471.data_ptr()))
    aten.bernoulli_(buf430, 0.9)
    buf433 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf434 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf435 = reinterpret_tensor(buf421, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf421  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_79(c_void_p(buf430.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()))
    buf436 = reinterpret_tensor(buf423, (24, 512, 64), (32768, 64, 1), 0); del buf423  # reuse
    # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf434, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf435, (24, 512, 64), (32768, 64, 1), 0), out=buf436)
    buf437 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_80(c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    buf438 = reinterpret_tensor(buf436, (512, 1536), (1536, 1), 0); del buf436  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf437, reinterpret_tensor(primals_171, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf438)
    del primals_172
    aten.bernoulli_(buf439, 0.9)
    buf442 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf443 = reinterpret_tensor(buf438, (1, 512, 1536), (786432, 1536, 1), 0); del buf438  # reuse
    buf444 = buf416; del buf416  # reuse
    buf445 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf447 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf448 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_81(c_void_p(buf443.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    del primals_164
    buf449 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_83], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf448, reinterpret_tensor(primals_175, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf449)
    del primals_176
    buf450 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_82(c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    buf451 = reinterpret_tensor(buf443, (512, 1536), (1536, 1), 0); del buf443  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf450, reinterpret_tensor(primals_177, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf451)
    del primals_178
    aten.bernoulli_(buf452, 0.9)
    buf455 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf456 = reinterpret_tensor(buf451, (1, 512, 1536), (786432, 1536, 1), 0); del buf451  # reuse
    buf457 = buf444; del buf444  # reuse
    buf458 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf460 = buf439; del buf439  # reuse
    buf461 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_83(c_void_p(buf456.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    del primals_174
    buf462 = reinterpret_tensor(buf456, (512, 1536), (1536, 1), 0); del buf456  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf461, reinterpret_tensor(primals_181, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf462)
    del primals_182
    buf463 = reinterpret_tensor(buf452, (512, 1536), (1536, 1), 0); del buf452  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf461, reinterpret_tensor(primals_183, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf463)
    del primals_184
    buf464 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_11_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_186, buf461, reinterpret_tensor(primals_185, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf464)
    del primals_186
    buf465 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf466 = reinterpret_tensor(buf463, (24, 64, 512), (64, 1, 1536), 0); del buf463  # reuse
    cpp_fused_clone_div_sqrt_84(c_void_p(buf466.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf465.data_ptr()))
    buf467 = reinterpret_tensor(buf430, (24, 512, 512), (262144, 512, 1), 0); del buf430  # reuse
    # Source Nodes: [attention_scores_33, scale, truediv_11], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf465, (24, 512, 64), (32768, 64, 1), 0), buf466, out=buf467)
    buf468 = buf427; del buf427  # reuse
    buf469 = reinterpret_tensor(buf467, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf467  # reuse
    buf470 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_85(c_void_p(buf469.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()))
    aten.bernoulli_(buf471, 0.9)
    buf474 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf475 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf476 = reinterpret_tensor(buf462, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf462  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_86(c_void_p(buf471.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    buf477 = reinterpret_tensor(buf464, (24, 512, 64), (32768, 64, 1), 0); del buf464  # reuse
    # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf475, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf476, (24, 512, 64), (32768, 64, 1), 0), out=buf477)
    buf478 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_87(c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    buf479 = reinterpret_tensor(buf477, (512, 1536), (1536, 1), 0); del buf477  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf478, reinterpret_tensor(primals_187, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf479)
    del primals_188
    buf480 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf493 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf521 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf534 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_88(c_void_p(buf4.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf534.data_ptr()))
    aten.bernoulli_(buf480, 0.9)
    buf483 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf484 = reinterpret_tensor(buf479, (1, 512, 1536), (786432, 1536, 1), 0); del buf479  # reuse
    buf485 = buf457; del buf457  # reuse
    buf486 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf488 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf489 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_89(c_void_p(buf484.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    del primals_180
    buf490 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_192, buf489, reinterpret_tensor(primals_191, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf490)
    del primals_192
    buf491 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_90(c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    buf492 = reinterpret_tensor(buf484, (512, 1536), (1536, 1), 0); del buf484  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, buf491, reinterpret_tensor(primals_193, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf492)
    del primals_194
    aten.bernoulli_(buf493, 0.9)
    buf496 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf497 = reinterpret_tensor(buf492, (1, 512, 1536), (786432, 1536, 1), 0); del buf492  # reuse
    buf498 = buf485; del buf485  # reuse
    buf499 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf501 = buf480; del buf480  # reuse
    buf502 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_91(c_void_p(buf497.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()))
    del primals_190
    buf503 = reinterpret_tensor(buf497, (512, 1536), (1536, 1), 0); del buf497  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_198, buf502, reinterpret_tensor(primals_197, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf503)
    del primals_198
    buf504 = reinterpret_tensor(buf493, (512, 1536), (1536, 1), 0); del buf493  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf502, reinterpret_tensor(primals_199, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf504)
    del primals_200
    buf505 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_12_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_202, buf502, reinterpret_tensor(primals_201, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf505)
    del primals_202
    buf506 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf507 = reinterpret_tensor(buf504, (24, 64, 512), (64, 1, 1536), 0); del buf504  # reuse
    cpp_fused_clone_div_sqrt_92(c_void_p(buf507.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf506.data_ptr()))
    buf508 = reinterpret_tensor(buf471, (24, 512, 512), (262144, 512, 1), 0); del buf471  # reuse
    # Source Nodes: [attention_scores_36, scale, truediv_12], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf506, (24, 512, 64), (32768, 64, 1), 0), buf507, out=buf508)
    buf509 = buf468; del buf468  # reuse
    buf510 = reinterpret_tensor(buf508, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf508  # reuse
    buf511 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf512 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf553 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_93(c_void_p(buf510.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf553.data_ptr()))
    aten.bernoulli_(buf512, 0.9)
    buf515 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf516 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf517 = reinterpret_tensor(buf503, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf503  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_94(c_void_p(buf512.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()))
    buf518 = reinterpret_tensor(buf505, (24, 512, 64), (32768, 64, 1), 0); del buf505  # reuse
    # Source Nodes: [context_layer_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf516, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf517, (24, 512, 64), (32768, 64, 1), 0), out=buf518)
    buf519 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_95(c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()))
    buf520 = reinterpret_tensor(buf518, (512, 1536), (1536, 1), 0); del buf518  # reuse
    # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf519, reinterpret_tensor(primals_203, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf520)
    del primals_204
    aten.bernoulli_(buf521, 0.9)
    buf524 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf525 = reinterpret_tensor(buf520, (1, 512, 1536), (786432, 1536, 1), 0); del buf520  # reuse
    buf526 = buf498; del buf498  # reuse
    buf527 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf529 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf530 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_96(c_void_p(buf525.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()))
    del primals_196
    buf531 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf530, reinterpret_tensor(primals_207, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf531)
    del primals_208
    buf532 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_97(c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = reinterpret_tensor(buf525, (512, 1536), (1536, 1), 0); del buf525  # reuse
    # Source Nodes: [hidden_states_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, buf532, reinterpret_tensor(primals_209, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf533)
    del primals_210
    aten.bernoulli_(buf534, 0.9)
    buf537 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf538 = reinterpret_tensor(buf533, (1, 512, 1536), (786432, 1536, 1), 0); del buf533  # reuse
    buf539 = buf526; del buf526  # reuse
    buf540 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf542 = buf521; del buf521  # reuse
    buf543 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_98(c_void_p(buf538.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()))
    del primals_206
    buf544 = reinterpret_tensor(buf538, (512, 1536), (1536, 1), 0); del buf538  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_214, buf543, reinterpret_tensor(primals_213, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf544)
    del primals_214
    buf545 = reinterpret_tensor(buf534, (512, 1536), (1536, 1), 0); del buf534  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_216, buf543, reinterpret_tensor(primals_215, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf545)
    del primals_216
    buf546 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_13_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_218, buf543, reinterpret_tensor(primals_217, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf546)
    del primals_218
    buf547 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf548 = reinterpret_tensor(buf545, (24, 64, 512), (64, 1, 1536), 0); del buf545  # reuse
    cpp_fused_clone_div_sqrt_99(c_void_p(buf548.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf547.data_ptr()))
    buf549 = reinterpret_tensor(buf512, (24, 512, 512), (262144, 512, 1), 0); del buf512  # reuse
    # Source Nodes: [attention_scores_39, scale, truediv_13], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf547, (24, 512, 64), (32768, 64, 1), 0), buf548, out=buf549)
    buf550 = buf509; del buf509  # reuse
    buf551 = reinterpret_tensor(buf549, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf549  # reuse
    buf552 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_100(c_void_p(buf551.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf552.data_ptr()))
    aten.bernoulli_(buf553, 0.9)
    buf556 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf557 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf558 = reinterpret_tensor(buf544, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf544  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_101(c_void_p(buf553.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    buf559 = reinterpret_tensor(buf546, (24, 512, 64), (32768, 64, 1), 0); del buf546  # reuse
    # Source Nodes: [context_layer_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf557, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf558, (24, 512, 64), (32768, 64, 1), 0), out=buf559)
    buf560 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_102(c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()))
    buf561 = reinterpret_tensor(buf559, (512, 1536), (1536, 1), 0); del buf559  # reuse
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_220, buf560, reinterpret_tensor(primals_219, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf561)
    del primals_220
    buf562 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf575 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf603 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf616 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_103(c_void_p(buf4.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf616.data_ptr()))
    aten.bernoulli_(buf562, 0.9)
    buf565 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf566 = reinterpret_tensor(buf561, (1, 512, 1536), (786432, 1536, 1), 0); del buf561  # reuse
    buf567 = buf539; del buf539  # reuse
    buf568 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf570 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf571 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_104(c_void_p(buf566.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()))
    del primals_212
    buf572 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_224, buf571, reinterpret_tensor(primals_223, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf572)
    del primals_224
    buf573 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_105(c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = reinterpret_tensor(buf566, (512, 1536), (1536, 1), 0); del buf566  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_226, buf573, reinterpret_tensor(primals_225, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf574)
    del primals_226
    aten.bernoulli_(buf575, 0.9)
    buf578 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf579 = reinterpret_tensor(buf574, (1, 512, 1536), (786432, 1536, 1), 0); del buf574  # reuse
    buf580 = buf567; del buf567  # reuse
    buf581 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf583 = buf562; del buf562  # reuse
    buf584 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_106(c_void_p(buf579.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()))
    del primals_222
    buf585 = reinterpret_tensor(buf579, (512, 1536), (1536, 1), 0); del buf579  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_230, buf584, reinterpret_tensor(primals_229, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf585)
    del primals_230
    buf586 = reinterpret_tensor(buf575, (512, 1536), (1536, 1), 0); del buf575  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_232, buf584, reinterpret_tensor(primals_231, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf586)
    del primals_232
    buf587 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_14_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_234, buf584, reinterpret_tensor(primals_233, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf587)
    del primals_234
    buf588 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf589 = reinterpret_tensor(buf586, (24, 64, 512), (64, 1, 1536), 0); del buf586  # reuse
    cpp_fused_clone_div_sqrt_107(c_void_p(buf589.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf588.data_ptr()))
    buf590 = reinterpret_tensor(buf553, (24, 512, 512), (262144, 512, 1), 0); del buf553  # reuse
    # Source Nodes: [attention_scores_42, scale, truediv_14], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf588, (24, 512, 64), (32768, 64, 1), 0), buf589, out=buf590)
    buf591 = buf550; del buf550  # reuse
    buf592 = reinterpret_tensor(buf590, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf590  # reuse
    buf593 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf594 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf635 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_108(c_void_p(buf592.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf635.data_ptr()))
    aten.bernoulli_(buf594, 0.9)
    buf597 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf598 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf599 = reinterpret_tensor(buf585, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf585  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_109(c_void_p(buf594.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()))
    buf600 = reinterpret_tensor(buf587, (24, 512, 64), (32768, 64, 1), 0); del buf587  # reuse
    # Source Nodes: [context_layer_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf598, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf599, (24, 512, 64), (32768, 64, 1), 0), out=buf600)
    buf601 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_110(c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()))
    buf602 = reinterpret_tensor(buf600, (512, 1536), (1536, 1), 0); del buf600  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_236, buf601, reinterpret_tensor(primals_235, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf602)
    del primals_236
    aten.bernoulli_(buf603, 0.9)
    buf606 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf607 = reinterpret_tensor(buf602, (1, 512, 1536), (786432, 1536, 1), 0); del buf602  # reuse
    buf608 = buf580; del buf580  # reuse
    buf609 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf611 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf612 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_111(c_void_p(buf607.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()))
    del primals_228
    buf613 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_240, buf612, reinterpret_tensor(primals_239, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf613)
    del primals_240
    buf614 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_112(c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()))
    buf615 = reinterpret_tensor(buf607, (512, 1536), (1536, 1), 0); del buf607  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_242, buf614, reinterpret_tensor(primals_241, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf615)
    del primals_242
    aten.bernoulli_(buf616, 0.9)
    buf619 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf620 = reinterpret_tensor(buf615, (1, 512, 1536), (786432, 1536, 1), 0); del buf615  # reuse
    buf621 = buf608; del buf608  # reuse
    buf622 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf624 = buf603; del buf603  # reuse
    buf625 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_113(c_void_p(buf620.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()))
    del primals_238
    buf626 = reinterpret_tensor(buf620, (512, 1536), (1536, 1), 0); del buf620  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_246, buf625, reinterpret_tensor(primals_245, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf626)
    del primals_246
    buf627 = reinterpret_tensor(buf616, (512, 1536), (1536, 1), 0); del buf616  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_248, buf625, reinterpret_tensor(primals_247, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf627)
    del primals_248
    buf628 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_15_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_250, buf625, reinterpret_tensor(primals_249, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf628)
    del primals_250
    buf629 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf630 = reinterpret_tensor(buf627, (24, 64, 512), (64, 1, 1536), 0); del buf627  # reuse
    cpp_fused_clone_div_sqrt_114(c_void_p(buf630.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf629.data_ptr()))
    buf631 = reinterpret_tensor(buf594, (24, 512, 512), (262144, 512, 1), 0); del buf594  # reuse
    # Source Nodes: [attention_scores_45, scale, truediv_15], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf629, (24, 512, 64), (32768, 64, 1), 0), buf630, out=buf631)
    buf632 = buf591; del buf591  # reuse
    buf633 = reinterpret_tensor(buf631, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf631  # reuse
    buf634 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_115(c_void_p(buf633.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()))
    aten.bernoulli_(buf635, 0.9)
    buf638 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf639 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf640 = reinterpret_tensor(buf626, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf626  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_116(c_void_p(buf635.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()))
    buf641 = reinterpret_tensor(buf628, (24, 512, 64), (32768, 64, 1), 0); del buf628  # reuse
    # Source Nodes: [context_layer_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf639, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf640, (24, 512, 64), (32768, 64, 1), 0), out=buf641)
    buf642 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_117(c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()))
    buf643 = reinterpret_tensor(buf641, (512, 1536), (1536, 1), 0); del buf641  # reuse
    # Source Nodes: [hidden_states_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_252, buf642, reinterpret_tensor(primals_251, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf643)
    del primals_252
    buf644 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf657 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf685 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf698 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_118(c_void_p(buf4.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf698.data_ptr()))
    aten.bernoulli_(buf644, 0.9)
    buf647 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf648 = reinterpret_tensor(buf643, (1, 512, 1536), (786432, 1536, 1), 0); del buf643  # reuse
    buf649 = buf621; del buf621  # reuse
    buf650 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf652 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf653 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_119(c_void_p(buf648.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()))
    del primals_244
    buf654 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, buf653, reinterpret_tensor(primals_255, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf654)
    del primals_256
    buf655 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_120(c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()))
    buf656 = reinterpret_tensor(buf648, (512, 1536), (1536, 1), 0); del buf648  # reuse
    # Source Nodes: [hidden_states_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_258, buf655, reinterpret_tensor(primals_257, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf656)
    del primals_258
    aten.bernoulli_(buf657, 0.9)
    buf660 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf661 = reinterpret_tensor(buf656, (1, 512, 1536), (786432, 1536, 1), 0); del buf656  # reuse
    buf662 = buf649; del buf649  # reuse
    buf663 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf665 = buf644; del buf644  # reuse
    buf666 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_121(c_void_p(buf661.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()))
    del primals_254
    buf667 = reinterpret_tensor(buf661, (512, 1536), (1536, 1), 0); del buf661  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf666, reinterpret_tensor(primals_261, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf667)
    del primals_262
    buf668 = reinterpret_tensor(buf657, (512, 1536), (1536, 1), 0); del buf657  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_264, buf666, reinterpret_tensor(primals_263, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf668)
    del primals_264
    buf669 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_16_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_266, buf666, reinterpret_tensor(primals_265, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf669)
    del primals_266
    buf670 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf671 = reinterpret_tensor(buf668, (24, 64, 512), (64, 1, 1536), 0); del buf668  # reuse
    cpp_fused_clone_div_sqrt_122(c_void_p(buf671.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf670.data_ptr()))
    buf672 = reinterpret_tensor(buf635, (24, 512, 512), (262144, 512, 1), 0); del buf635  # reuse
    # Source Nodes: [attention_scores_48, scale, truediv_16], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf670, (24, 512, 64), (32768, 64, 1), 0), buf671, out=buf672)
    buf673 = buf632; del buf632  # reuse
    buf674 = reinterpret_tensor(buf672, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf672  # reuse
    buf675 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf676 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf717 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_123(c_void_p(buf674.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf717.data_ptr()))
    aten.bernoulli_(buf676, 0.9)
    buf679 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf680 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf681 = reinterpret_tensor(buf667, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf667  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_124(c_void_p(buf676.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()))
    buf682 = reinterpret_tensor(buf669, (24, 512, 64), (32768, 64, 1), 0); del buf669  # reuse
    # Source Nodes: [context_layer_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf680, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf681, (24, 512, 64), (32768, 64, 1), 0), out=buf682)
    buf683 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_125(c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()))
    buf684 = reinterpret_tensor(buf682, (512, 1536), (1536, 1), 0); del buf682  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_268, buf683, reinterpret_tensor(primals_267, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf684)
    del primals_268
    aten.bernoulli_(buf685, 0.9)
    buf688 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf689 = reinterpret_tensor(buf684, (1, 512, 1536), (786432, 1536, 1), 0); del buf684  # reuse
    buf690 = buf662; del buf662  # reuse
    buf691 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf693 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf694 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_126(c_void_p(buf689.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()))
    del primals_260
    buf695 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_272, buf694, reinterpret_tensor(primals_271, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf695)
    del primals_272
    buf696 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_127(c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()))
    buf697 = reinterpret_tensor(buf689, (512, 1536), (1536, 1), 0); del buf689  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_274, buf696, reinterpret_tensor(primals_273, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf697)
    del primals_274
    aten.bernoulli_(buf698, 0.9)
    buf701 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf702 = reinterpret_tensor(buf697, (1, 512, 1536), (786432, 1536, 1), 0); del buf697  # reuse
    buf703 = buf690; del buf690  # reuse
    buf704 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf706 = buf685; del buf685  # reuse
    buf707 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_128(c_void_p(buf702.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf707.data_ptr()))
    del primals_270
    buf708 = reinterpret_tensor(buf702, (512, 1536), (1536, 1), 0); del buf702  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_278, buf707, reinterpret_tensor(primals_277, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf708)
    del primals_278
    buf709 = reinterpret_tensor(buf698, (512, 1536), (1536, 1), 0); del buf698  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_280, buf707, reinterpret_tensor(primals_279, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf709)
    del primals_280
    buf710 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_17_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_282, buf707, reinterpret_tensor(primals_281, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf710)
    del primals_282
    buf711 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf712 = reinterpret_tensor(buf709, (24, 64, 512), (64, 1, 1536), 0); del buf709  # reuse
    cpp_fused_clone_div_sqrt_129(c_void_p(buf712.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf711.data_ptr()))
    buf713 = reinterpret_tensor(buf676, (24, 512, 512), (262144, 512, 1), 0); del buf676  # reuse
    # Source Nodes: [attention_scores_51, scale, truediv_17], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf711, (24, 512, 64), (32768, 64, 1), 0), buf712, out=buf713)
    buf714 = buf673; del buf673  # reuse
    buf715 = reinterpret_tensor(buf713, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf713  # reuse
    buf716 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_130(c_void_p(buf715.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf716.data_ptr()))
    aten.bernoulli_(buf717, 0.9)
    buf720 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf721 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf722 = reinterpret_tensor(buf708, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf708  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_131(c_void_p(buf717.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()))
    buf723 = reinterpret_tensor(buf710, (24, 512, 64), (32768, 64, 1), 0); del buf710  # reuse
    # Source Nodes: [context_layer_51], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf721, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf722, (24, 512, 64), (32768, 64, 1), 0), out=buf723)
    buf724 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_132(c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()))
    buf725 = reinterpret_tensor(buf723, (512, 1536), (1536, 1), 0); del buf723  # reuse
    # Source Nodes: [hidden_states_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_284, buf724, reinterpret_tensor(primals_283, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf725)
    del primals_284
    buf726 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf739 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf767 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf780 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_133(c_void_p(buf4.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf780.data_ptr()))
    aten.bernoulli_(buf726, 0.9)
    buf729 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf730 = reinterpret_tensor(buf725, (1, 512, 1536), (786432, 1536, 1), 0); del buf725  # reuse
    buf731 = buf703; del buf703  # reuse
    buf732 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf734 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf735 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_134(c_void_p(buf730.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf735.data_ptr()))
    del primals_276
    buf736 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_288, buf735, reinterpret_tensor(primals_287, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf736)
    del primals_288
    buf737 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_135(c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()))
    buf738 = reinterpret_tensor(buf730, (512, 1536), (1536, 1), 0); del buf730  # reuse
    # Source Nodes: [hidden_states_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_290, buf737, reinterpret_tensor(primals_289, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf738)
    del primals_290
    aten.bernoulli_(buf739, 0.9)
    buf742 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf743 = reinterpret_tensor(buf738, (1, 512, 1536), (786432, 1536, 1), 0); del buf738  # reuse
    buf744 = buf731; del buf731  # reuse
    buf745 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf747 = buf726; del buf726  # reuse
    buf748 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_136(c_void_p(buf743.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()))
    del primals_286
    buf749 = reinterpret_tensor(buf743, (512, 1536), (1536, 1), 0); del buf743  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_294, buf748, reinterpret_tensor(primals_293, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf749)
    del primals_294
    buf750 = reinterpret_tensor(buf739, (512, 1536), (1536, 1), 0); del buf739  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_296, buf748, reinterpret_tensor(primals_295, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf750)
    del primals_296
    buf751 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_18_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_298, buf748, reinterpret_tensor(primals_297, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf751)
    del primals_298
    buf752 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf753 = reinterpret_tensor(buf750, (24, 64, 512), (64, 1, 1536), 0); del buf750  # reuse
    cpp_fused_clone_div_sqrt_137(c_void_p(buf753.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf752.data_ptr()))
    buf754 = reinterpret_tensor(buf717, (24, 512, 512), (262144, 512, 1), 0); del buf717  # reuse
    # Source Nodes: [attention_scores_54, scale, truediv_18], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf752, (24, 512, 64), (32768, 64, 1), 0), buf753, out=buf754)
    buf755 = buf714; del buf714  # reuse
    buf756 = reinterpret_tensor(buf754, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf754  # reuse
    buf757 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf758 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf799 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_138(c_void_p(buf756.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf799.data_ptr()))
    aten.bernoulli_(buf758, 0.9)
    buf761 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf762 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf763 = reinterpret_tensor(buf749, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf749  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_139(c_void_p(buf758.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf763.data_ptr()))
    buf764 = reinterpret_tensor(buf751, (24, 512, 64), (32768, 64, 1), 0); del buf751  # reuse
    # Source Nodes: [context_layer_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf762, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf763, (24, 512, 64), (32768, 64, 1), 0), out=buf764)
    buf765 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_140(c_void_p(buf764.data_ptr()), c_void_p(buf765.data_ptr()))
    buf766 = reinterpret_tensor(buf764, (512, 1536), (1536, 1), 0); del buf764  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_300, buf765, reinterpret_tensor(primals_299, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf766)
    del primals_300
    aten.bernoulli_(buf767, 0.9)
    buf770 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf771 = reinterpret_tensor(buf766, (1, 512, 1536), (786432, 1536, 1), 0); del buf766  # reuse
    buf772 = buf744; del buf744  # reuse
    buf773 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf775 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf776 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_141(c_void_p(buf771.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()))
    del primals_292
    buf777 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_304, buf776, reinterpret_tensor(primals_303, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf777)
    del primals_304
    buf778 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_142(c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()))
    buf779 = reinterpret_tensor(buf771, (512, 1536), (1536, 1), 0); del buf771  # reuse
    # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_306, buf778, reinterpret_tensor(primals_305, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf779)
    del primals_306
    aten.bernoulli_(buf780, 0.9)
    buf783 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf784 = reinterpret_tensor(buf779, (1, 512, 1536), (786432, 1536, 1), 0); del buf779  # reuse
    buf785 = buf772; del buf772  # reuse
    buf786 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf788 = buf767; del buf767  # reuse
    buf789 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_143(c_void_p(buf784.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(buf789.data_ptr()))
    del primals_302
    buf790 = reinterpret_tensor(buf784, (512, 1536), (1536, 1), 0); del buf784  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_310, buf789, reinterpret_tensor(primals_309, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf790)
    del primals_310
    buf791 = reinterpret_tensor(buf780, (512, 1536), (1536, 1), 0); del buf780  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_312, buf789, reinterpret_tensor(primals_311, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf791)
    del primals_312
    buf792 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_19_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_314, buf789, reinterpret_tensor(primals_313, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf792)
    del primals_314
    buf793 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf794 = reinterpret_tensor(buf791, (24, 64, 512), (64, 1, 1536), 0); del buf791  # reuse
    cpp_fused_clone_div_sqrt_144(c_void_p(buf794.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf793.data_ptr()))
    buf795 = reinterpret_tensor(buf758, (24, 512, 512), (262144, 512, 1), 0); del buf758  # reuse
    # Source Nodes: [attention_scores_57, scale, truediv_19], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf793, (24, 512, 64), (32768, 64, 1), 0), buf794, out=buf795)
    buf796 = buf755; del buf755  # reuse
    buf797 = reinterpret_tensor(buf795, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf795  # reuse
    buf798 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_145(c_void_p(buf797.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf798.data_ptr()))
    aten.bernoulli_(buf799, 0.9)
    buf802 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf803 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf804 = reinterpret_tensor(buf790, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf790  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_146(c_void_p(buf799.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(buf804.data_ptr()))
    buf805 = reinterpret_tensor(buf792, (24, 512, 64), (32768, 64, 1), 0); del buf792  # reuse
    # Source Nodes: [context_layer_57], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf803, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf804, (24, 512, 64), (32768, 64, 1), 0), out=buf805)
    buf806 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_147(c_void_p(buf805.data_ptr()), c_void_p(buf806.data_ptr()))
    buf807 = reinterpret_tensor(buf805, (512, 1536), (1536, 1), 0); del buf805  # reuse
    # Source Nodes: [hidden_states_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_316, buf806, reinterpret_tensor(primals_315, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf807)
    del primals_316
    buf808 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf821 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf849 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf862 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_148(c_void_p(buf4.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf862.data_ptr()))
    aten.bernoulli_(buf808, 0.9)
    buf811 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf812 = reinterpret_tensor(buf807, (1, 512, 1536), (786432, 1536, 1), 0); del buf807  # reuse
    buf813 = buf785; del buf785  # reuse
    buf814 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf816 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf817 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_149(c_void_p(buf812.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf817.data_ptr()))
    del primals_308
    buf818 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_320, buf817, reinterpret_tensor(primals_319, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf818)
    del primals_320
    buf819 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_150(c_void_p(buf818.data_ptr()), c_void_p(buf819.data_ptr()))
    buf820 = reinterpret_tensor(buf812, (512, 1536), (1536, 1), 0); del buf812  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_322, buf819, reinterpret_tensor(primals_321, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf820)
    del primals_322
    aten.bernoulli_(buf821, 0.9)
    buf824 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf825 = reinterpret_tensor(buf820, (1, 512, 1536), (786432, 1536, 1), 0); del buf820  # reuse
    buf826 = buf813; del buf813  # reuse
    buf827 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf829 = buf808; del buf808  # reuse
    buf830 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_151(c_void_p(buf825.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf830.data_ptr()))
    del primals_318
    buf831 = reinterpret_tensor(buf825, (512, 1536), (1536, 1), 0); del buf825  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_326, buf830, reinterpret_tensor(primals_325, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf831)
    del primals_326
    buf832 = reinterpret_tensor(buf821, (512, 1536), (1536, 1), 0); del buf821  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_328, buf830, reinterpret_tensor(primals_327, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf832)
    del primals_328
    buf833 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_20_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_330, buf830, reinterpret_tensor(primals_329, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf833)
    del primals_330
    buf834 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf835 = reinterpret_tensor(buf832, (24, 64, 512), (64, 1, 1536), 0); del buf832  # reuse
    cpp_fused_clone_div_sqrt_152(c_void_p(buf835.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf834.data_ptr()))
    buf836 = reinterpret_tensor(buf799, (24, 512, 512), (262144, 512, 1), 0); del buf799  # reuse
    # Source Nodes: [attention_scores_60, scale, truediv_20], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf834, (24, 512, 64), (32768, 64, 1), 0), buf835, out=buf836)
    buf837 = buf796; del buf796  # reuse
    buf838 = reinterpret_tensor(buf836, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf836  # reuse
    buf839 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf840 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf881 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_153(c_void_p(buf838.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf881.data_ptr()))
    aten.bernoulli_(buf840, 0.9)
    buf843 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf844 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf845 = reinterpret_tensor(buf831, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf831  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_154(c_void_p(buf840.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf845.data_ptr()))
    buf846 = reinterpret_tensor(buf833, (24, 512, 64), (32768, 64, 1), 0); del buf833  # reuse
    # Source Nodes: [context_layer_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf844, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf845, (24, 512, 64), (32768, 64, 1), 0), out=buf846)
    buf847 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_155(c_void_p(buf846.data_ptr()), c_void_p(buf847.data_ptr()))
    buf848 = reinterpret_tensor(buf846, (512, 1536), (1536, 1), 0); del buf846  # reuse
    # Source Nodes: [hidden_states_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_332, buf847, reinterpret_tensor(primals_331, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf848)
    del primals_332
    aten.bernoulli_(buf849, 0.9)
    buf852 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf853 = reinterpret_tensor(buf848, (1, 512, 1536), (786432, 1536, 1), 0); del buf848  # reuse
    buf854 = buf826; del buf826  # reuse
    buf855 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf857 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf858 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_156(c_void_p(buf853.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(buf858.data_ptr()))
    del primals_324
    buf859 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_336, buf858, reinterpret_tensor(primals_335, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf859)
    del primals_336
    buf860 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_157(c_void_p(buf859.data_ptr()), c_void_p(buf860.data_ptr()))
    buf861 = reinterpret_tensor(buf853, (512, 1536), (1536, 1), 0); del buf853  # reuse
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_338, buf860, reinterpret_tensor(primals_337, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf861)
    del primals_338
    aten.bernoulli_(buf862, 0.9)
    buf865 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf866 = reinterpret_tensor(buf861, (1, 512, 1536), (786432, 1536, 1), 0); del buf861  # reuse
    buf867 = buf854; del buf854  # reuse
    buf868 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf870 = buf849; del buf849  # reuse
    buf871 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_158(c_void_p(buf866.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf867.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf871.data_ptr()))
    del primals_334
    buf872 = reinterpret_tensor(buf866, (512, 1536), (1536, 1), 0); del buf866  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_342, buf871, reinterpret_tensor(primals_341, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf872)
    del primals_342
    buf873 = reinterpret_tensor(buf862, (512, 1536), (1536, 1), 0); del buf862  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_344, buf871, reinterpret_tensor(primals_343, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf873)
    del primals_344
    buf874 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_21_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_346, buf871, reinterpret_tensor(primals_345, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf874)
    del primals_346
    buf875 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf876 = reinterpret_tensor(buf873, (24, 64, 512), (64, 1, 1536), 0); del buf873  # reuse
    cpp_fused_clone_div_sqrt_159(c_void_p(buf876.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf875.data_ptr()))
    buf877 = reinterpret_tensor(buf840, (24, 512, 512), (262144, 512, 1), 0); del buf840  # reuse
    # Source Nodes: [attention_scores_63, scale, truediv_21], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf875, (24, 512, 64), (32768, 64, 1), 0), buf876, out=buf877)
    buf878 = buf837; del buf837  # reuse
    buf879 = reinterpret_tensor(buf877, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf877  # reuse
    buf880 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_160(c_void_p(buf879.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf880.data_ptr()))
    aten.bernoulli_(buf881, 0.9)
    buf884 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf885 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf886 = reinterpret_tensor(buf872, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf872  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_161(c_void_p(buf881.data_ptr()), c_void_p(buf879.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf886.data_ptr()))
    buf887 = reinterpret_tensor(buf874, (24, 512, 64), (32768, 64, 1), 0); del buf874  # reuse
    # Source Nodes: [context_layer_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf885, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf886, (24, 512, 64), (32768, 64, 1), 0), out=buf887)
    buf888 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_162(c_void_p(buf887.data_ptr()), c_void_p(buf888.data_ptr()))
    buf889 = reinterpret_tensor(buf887, (512, 1536), (1536, 1), 0); del buf887  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_348, buf888, reinterpret_tensor(primals_347, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf889)
    del primals_348
    buf890 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf903 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf931 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf944 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_163(c_void_p(buf4.data_ptr()), c_void_p(buf890.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf944.data_ptr()))
    aten.bernoulli_(buf890, 0.9)
    buf893 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf894 = reinterpret_tensor(buf889, (1, 512, 1536), (786432, 1536, 1), 0); del buf889  # reuse
    buf895 = buf867; del buf867  # reuse
    buf896 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf898 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf899 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_164(c_void_p(buf894.data_ptr()), c_void_p(buf890.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf898.data_ptr()), c_void_p(buf899.data_ptr()))
    del primals_340
    buf900 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_171], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_352, buf899, reinterpret_tensor(primals_351, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf900)
    del primals_352
    buf901 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_165(c_void_p(buf900.data_ptr()), c_void_p(buf901.data_ptr()))
    buf902 = reinterpret_tensor(buf894, (512, 1536), (1536, 1), 0); del buf894  # reuse
    # Source Nodes: [hidden_states_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_354, buf901, reinterpret_tensor(primals_353, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf902)
    del primals_354
    aten.bernoulli_(buf903, 0.9)
    buf906 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf907 = reinterpret_tensor(buf902, (1, 512, 1536), (786432, 1536, 1), 0); del buf902  # reuse
    buf908 = buf895; del buf895  # reuse
    buf909 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf911 = buf890; del buf890  # reuse
    buf912 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_166(c_void_p(buf907.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(buf898.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf908.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(buf912.data_ptr()))
    del primals_350
    buf913 = reinterpret_tensor(buf907, (512, 1536), (1536, 1), 0); del buf907  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_358, buf912, reinterpret_tensor(primals_357, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf913)
    del primals_358
    buf914 = reinterpret_tensor(buf903, (512, 1536), (1536, 1), 0); del buf903  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_360, buf912, reinterpret_tensor(primals_359, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf914)
    del primals_360
    buf915 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_22_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_362, buf912, reinterpret_tensor(primals_361, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf915)
    del primals_362
    buf916 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf917 = reinterpret_tensor(buf914, (24, 64, 512), (64, 1, 1536), 0); del buf914  # reuse
    cpp_fused_clone_div_sqrt_167(c_void_p(buf917.data_ptr()), c_void_p(buf913.data_ptr()), c_void_p(buf916.data_ptr()))
    buf918 = reinterpret_tensor(buf881, (24, 512, 512), (262144, 512, 1), 0); del buf881  # reuse
    # Source Nodes: [attention_scores_66, scale, truediv_22], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf916, (24, 512, 64), (32768, 64, 1), 0), buf917, out=buf918)
    buf919 = buf878; del buf878  # reuse
    buf920 = reinterpret_tensor(buf918, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf918  # reuse
    buf921 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf922 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf963 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bernoulli_bitwise_not_lift_fresh_masked_fill_168(c_void_p(buf920.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf922.data_ptr()), c_void_p(buf963.data_ptr()))
    aten.bernoulli_(buf922, 0.9)
    buf925 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf926 = buf19; del buf19  # reuse
    buf927 = reinterpret_tensor(buf913, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf913  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_masked_fill_mul_rsub_169(c_void_p(buf922.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf915.data_ptr()), c_void_p(buf925.data_ptr()), c_void_p(buf926.data_ptr()), c_void_p(buf927.data_ptr()))
    buf928 = reinterpret_tensor(buf915, (24, 512, 64), (32768, 64, 1), 0); del buf915  # reuse
    # Source Nodes: [context_layer_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf926, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf927, (24, 512, 64), (32768, 64, 1), 0), out=buf928)
    buf929 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_170(c_void_p(buf928.data_ptr()), c_void_p(buf929.data_ptr()))
    buf930 = reinterpret_tensor(buf928, (512, 1536), (1536, 1), 0); del buf928  # reuse
    # Source Nodes: [hidden_states_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_364, buf929, reinterpret_tensor(primals_363, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf930)
    del primals_364
    aten.bernoulli_(buf931, 0.9)
    buf934 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf935 = reinterpret_tensor(buf930, (1, 512, 1536), (786432, 1536, 1), 0); del buf930  # reuse
    buf936 = buf908; del buf908  # reuse
    buf937 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf939 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf940 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_171(c_void_p(buf935.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf940.data_ptr()))
    del primals_356
    buf941 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_368, buf940, reinterpret_tensor(primals_367, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf941)
    del primals_368
    buf942 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_172(c_void_p(buf941.data_ptr()), c_void_p(buf942.data_ptr()))
    buf943 = reinterpret_tensor(buf935, (512, 1536), (1536, 1), 0); del buf935  # reuse
    # Source Nodes: [hidden_states_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_370, buf942, reinterpret_tensor(primals_369, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf943)
    del primals_370
    aten.bernoulli_(buf944, 0.9)
    buf947 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf948 = reinterpret_tensor(buf943, (1, 512, 1536), (786432, 1536, 1), 0); del buf943  # reuse
    buf949 = buf936; del buf936  # reuse
    buf950 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf952 = buf931; del buf931  # reuse
    buf953 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_173(c_void_p(buf948.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(buf947.data_ptr()), c_void_p(buf949.data_ptr()), c_void_p(buf950.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(buf953.data_ptr()))
    del primals_366
    buf954 = reinterpret_tensor(buf948, (512, 1536), (1536, 1), 0); del buf948  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_query_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_374, buf953, reinterpret_tensor(primals_373, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf954)
    del primals_374
    buf955 = reinterpret_tensor(buf944, (512, 1536), (1536, 1), 0); del buf944  # reuse
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_key_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_376, buf953, reinterpret_tensor(primals_375, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf955)
    del primals_376
    buf956 = empty((512, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___deberta_encoder_layer_23_attention_self_value_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_378, buf953, reinterpret_tensor(primals_377, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf956)
    del primals_378
    buf957 = empty((1, 24, 512, 64), device='cpu', dtype=torch.float32)
    buf958 = reinterpret_tensor(buf955, (24, 64, 512), (64, 1, 1536), 0); del buf955  # reuse
    cpp_fused_clone_div_sqrt_174(c_void_p(buf958.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf957.data_ptr()))
    buf959 = reinterpret_tensor(buf922, (24, 512, 512), (262144, 512, 1), 0); del buf922  # reuse
    # Source Nodes: [attention_scores_69, scale, truediv_23], Original ATen: [aten.bmm, aten.div, aten.sqrt]
    extern_kernels.bmm(reinterpret_tensor(buf957, (24, 512, 64), (32768, 64, 1), 0), buf958, out=buf959)
    buf960 = buf919; del buf919  # reuse
    buf961 = reinterpret_tensor(buf959, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf959  # reuse
    buf962 = empty_strided((1, 24, 512, 1), (12288, 512, 1, 12288), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_bitwise_not_lift_fresh_masked_fill_175(c_void_p(buf961.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf962.data_ptr()))
    del buf960
    aten.bernoulli_(buf963, 0.9)
    buf966 = empty((1, 24, 512, 512), device='cpu', dtype=torch.bool)
    buf967 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf1012 = empty((1, 24, 512, 512), device='cpu', dtype=torch.float32)
    buf968 = reinterpret_tensor(buf954, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf954  # reuse
    cpp_fused__softmax__to_copy_bitwise_not_clone_detach_masked_fill_mul_rsub_176(c_void_p(buf963.data_ptr()), c_void_p(buf961.data_ptr()), c_void_p(buf962.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf967.data_ptr()), c_void_p(buf1012.data_ptr()), c_void_p(buf968.data_ptr()))
    del buf961
    del buf962
    del buf963
    buf969 = reinterpret_tensor(buf956, (24, 512, 64), (32768, 64, 1), 0); del buf956  # reuse
    # Source Nodes: [context_layer_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf967, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf968, (24, 512, 64), (32768, 64, 1), 0), out=buf969)
    buf970 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_view_177(c_void_p(buf969.data_ptr()), c_void_p(buf970.data_ptr()))
    buf971 = reinterpret_tensor(buf969, (512, 1536), (1536, 1), 0); del buf969  # reuse
    # Source Nodes: [hidden_states_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_380, buf970, reinterpret_tensor(primals_379, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf971)
    del primals_380
    buf972 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    buf985 = empty((1, 512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_178(c_void_p(buf4.data_ptr()), c_void_p(buf972.data_ptr()), c_void_p(buf985.data_ptr()))
    aten.bernoulli_(buf972, 0.9)
    buf975 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf976 = reinterpret_tensor(buf971, (1, 512, 1536), (786432, 1536, 1), 0); del buf971  # reuse
    buf977 = buf949; del buf949  # reuse
    buf978 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf980 = buf4; del buf4  # reuse
    buf981 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_179(c_void_p(buf976.data_ptr()), c_void_p(buf972.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf977.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(buf981.data_ptr()))
    del primals_372
    buf982 = empty((512, 6144), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_384, buf981, reinterpret_tensor(primals_383, (1536, 6144), (1, 1536), 0), alpha=1, beta=1, out=buf982)
    del primals_384
    buf983 = empty((512, 6144), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_180(c_void_p(buf982.data_ptr()), c_void_p(buf983.data_ptr()))
    buf984 = reinterpret_tensor(buf976, (512, 1536), (1536, 1), 0); del buf976  # reuse
    # Source Nodes: [hidden_states_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_386, buf983, reinterpret_tensor(primals_385, (6144, 1536), (1, 6144), 0), alpha=1, beta=1, out=buf984)
    del primals_386
    aten.bernoulli_(buf985, 0.9)
    buf988 = empty((1, 512, 1536), device='cpu', dtype=torch.bool)
    buf989 = reinterpret_tensor(buf984, (1, 512, 1536), (786432, 1536, 1), 0); del buf984  # reuse
    buf990 = buf977; del buf977  # reuse
    buf991 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf993 = buf972; del buf972  # reuse
    buf994 = empty((512, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_mul_native_layer_norm_rsub_view_181(c_void_p(buf989.data_ptr()), c_void_p(buf985.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(buf988.data_ptr()), c_void_p(buf990.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(buf993.data_ptr()), c_void_p(buf994.data_ptr()))
    del primals_382
    del primals_388
    buf995 = empty((512, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_390, buf994, reinterpret_tensor(primals_389, (1536, 2), (1, 1536), 0), alpha=1, beta=1, out=buf995)
    del primals_390
    buf996 = reinterpret_tensor(buf990, (1, 512), (512, 1), 0); del buf990  # reuse
    buf998 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf997 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf1002 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf999 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf1000 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf1003 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf1004 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf1001 = empty((1, ), device='cpu', dtype=torch.bool)
    buf1005 = empty((1, ), device='cpu', dtype=torch.bool)
    buf1107 = empty((), device='cpu', dtype=torch.float32)
    buf1006 = empty((1, 1), device='cpu', dtype=torch.bool)
    buf1007 = empty((1, 1), device='cpu', dtype=torch.int64)
    buf1008 = empty((1, 1), device='cpu', dtype=torch.bool)
    buf1009 = empty((1, 1), device='cpu', dtype=torch.int64)
    buf1010 = reinterpret_tensor(buf991, (1, 512, 1), (512, 1, 1), 0); del buf991  # reuse
    buf1011 = reinterpret_tensor(buf978, (1, 512, 1), (512, 1, 1), 0); del buf978  # reuse
    buf1013 = reinterpret_tensor(buf989, (24, 512, 64), (32768, 64, 1), 0); del buf989  # reuse
    buf1014 = reinterpret_tensor(buf950, (1, 512, 1), (512, 1, 1), 0); del buf950  # reuse
    buf1015 = reinterpret_tensor(buf937, (1, 512, 1), (512, 1, 1), 0); del buf937  # reuse
    buf1016 = buf920; del buf920  # reuse
    buf1017 = reinterpret_tensor(buf985, (24, 512, 64), (32768, 64, 1), 0); del buf985  # reuse
    buf1018 = reinterpret_tensor(buf909, (1, 512, 1), (512, 1, 1), 0); del buf909  # reuse
    buf1019 = reinterpret_tensor(buf896, (1, 512, 1), (512, 1, 1), 0); del buf896  # reuse
    buf1020 = buf879; del buf879  # reuse
    buf1021 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1022 = reinterpret_tensor(buf868, (1, 512, 1), (512, 1, 1), 0); del buf868  # reuse
    buf1023 = reinterpret_tensor(buf855, (1, 512, 1), (512, 1, 1), 0); del buf855  # reuse
    buf1024 = buf838; del buf838  # reuse
    buf1025 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1026 = reinterpret_tensor(buf827, (1, 512, 1), (512, 1, 1), 0); del buf827  # reuse
    buf1027 = reinterpret_tensor(buf814, (1, 512, 1), (512, 1, 1), 0); del buf814  # reuse
    buf1028 = buf797; del buf797  # reuse
    buf1029 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1030 = reinterpret_tensor(buf786, (1, 512, 1), (512, 1, 1), 0); del buf786  # reuse
    buf1031 = reinterpret_tensor(buf773, (1, 512, 1), (512, 1, 1), 0); del buf773  # reuse
    buf1032 = buf756; del buf756  # reuse
    buf1033 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1034 = reinterpret_tensor(buf745, (1, 512, 1), (512, 1, 1), 0); del buf745  # reuse
    buf1035 = reinterpret_tensor(buf732, (1, 512, 1), (512, 1, 1), 0); del buf732  # reuse
    buf1036 = buf715; del buf715  # reuse
    buf1037 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1038 = reinterpret_tensor(buf704, (1, 512, 1), (512, 1, 1), 0); del buf704  # reuse
    buf1039 = reinterpret_tensor(buf691, (1, 512, 1), (512, 1, 1), 0); del buf691  # reuse
    buf1040 = buf674; del buf674  # reuse
    buf1041 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1042 = reinterpret_tensor(buf663, (1, 512, 1), (512, 1, 1), 0); del buf663  # reuse
    buf1043 = reinterpret_tensor(buf650, (1, 512, 1), (512, 1, 1), 0); del buf650  # reuse
    buf1044 = buf633; del buf633  # reuse
    buf1045 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1046 = reinterpret_tensor(buf622, (1, 512, 1), (512, 1, 1), 0); del buf622  # reuse
    buf1047 = reinterpret_tensor(buf609, (1, 512, 1), (512, 1, 1), 0); del buf609  # reuse
    buf1048 = buf592; del buf592  # reuse
    buf1049 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1050 = reinterpret_tensor(buf581, (1, 512, 1), (512, 1, 1), 0); del buf581  # reuse
    buf1051 = reinterpret_tensor(buf568, (1, 512, 1), (512, 1, 1), 0); del buf568  # reuse
    buf1052 = buf551; del buf551  # reuse
    buf1053 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1054 = reinterpret_tensor(buf540, (1, 512, 1), (512, 1, 1), 0); del buf540  # reuse
    buf1055 = reinterpret_tensor(buf527, (1, 512, 1), (512, 1, 1), 0); del buf527  # reuse
    buf1056 = buf510; del buf510  # reuse
    buf1057 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1058 = reinterpret_tensor(buf499, (1, 512, 1), (512, 1, 1), 0); del buf499  # reuse
    buf1059 = reinterpret_tensor(buf486, (1, 512, 1), (512, 1, 1), 0); del buf486  # reuse
    buf1060 = buf469; del buf469  # reuse
    buf1061 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1062 = reinterpret_tensor(buf458, (1, 512, 1), (512, 1, 1), 0); del buf458  # reuse
    buf1063 = reinterpret_tensor(buf445, (1, 512, 1), (512, 1, 1), 0); del buf445  # reuse
    buf1064 = buf428; del buf428  # reuse
    buf1065 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1066 = reinterpret_tensor(buf417, (1, 512, 1), (512, 1, 1), 0); del buf417  # reuse
    buf1067 = reinterpret_tensor(buf404, (1, 512, 1), (512, 1, 1), 0); del buf404  # reuse
    buf1068 = buf387; del buf387  # reuse
    buf1069 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1070 = reinterpret_tensor(buf376, (1, 512, 1), (512, 1, 1), 0); del buf376  # reuse
    buf1071 = reinterpret_tensor(buf363, (1, 512, 1), (512, 1, 1), 0); del buf363  # reuse
    buf1072 = buf346; del buf346  # reuse
    buf1073 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1074 = reinterpret_tensor(buf335, (1, 512, 1), (512, 1, 1), 0); del buf335  # reuse
    buf1075 = reinterpret_tensor(buf322, (1, 512, 1), (512, 1, 1), 0); del buf322  # reuse
    buf1076 = buf305; del buf305  # reuse
    buf1077 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1078 = reinterpret_tensor(buf294, (1, 512, 1), (512, 1, 1), 0); del buf294  # reuse
    buf1079 = reinterpret_tensor(buf281, (1, 512, 1), (512, 1, 1), 0); del buf281  # reuse
    buf1080 = buf264; del buf264  # reuse
    buf1081 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1082 = reinterpret_tensor(buf253, (1, 512, 1), (512, 1, 1), 0); del buf253  # reuse
    buf1083 = reinterpret_tensor(buf240, (1, 512, 1), (512, 1, 1), 0); del buf240  # reuse
    buf1084 = buf223; del buf223  # reuse
    buf1085 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1086 = reinterpret_tensor(buf212, (1, 512, 1), (512, 1, 1), 0); del buf212  # reuse
    buf1087 = reinterpret_tensor(buf199, (1, 512, 1), (512, 1, 1), 0); del buf199  # reuse
    buf1088 = buf182; del buf182  # reuse
    buf1089 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1090 = reinterpret_tensor(buf171, (1, 512, 1), (512, 1, 1), 0); del buf171  # reuse
    buf1091 = reinterpret_tensor(buf158, (1, 512, 1), (512, 1, 1), 0); del buf158  # reuse
    buf1092 = buf141; del buf141  # reuse
    buf1093 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1094 = reinterpret_tensor(buf130, (1, 512, 1), (512, 1, 1), 0); del buf130  # reuse
    buf1095 = reinterpret_tensor(buf117, (1, 512, 1), (512, 1, 1), 0); del buf117  # reuse
    buf1096 = buf100; del buf100  # reuse
    buf1097 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1098 = reinterpret_tensor(buf89, (1, 512, 1), (512, 1, 1), 0); del buf89  # reuse
    buf1099 = reinterpret_tensor(buf76, (1, 512, 1), (512, 1, 1), 0); del buf76  # reuse
    buf1100 = buf59; del buf59  # reuse
    buf1101 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1102 = reinterpret_tensor(buf48, (1, 512, 1), (512, 1, 1), 0); del buf48  # reuse
    buf1103 = reinterpret_tensor(buf35, (1, 512, 1), (512, 1, 1), 0); del buf35  # reuse
    buf1104 = buf17; del buf17  # reuse
    buf1105 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    buf1106 = reinterpret_tensor(buf1, (1, 512, 1), (512, 1, 1), 0); del buf1  # reuse
    cpp_fused__log_softmax__softmax_add_bitwise_not_clamp_clone_detach_div_embedding_masked_fill_native_layer_norm_native_layer_norm_backward_nll_loss_backward_nll_loss_forward_transpose_182(c_void_p(buf1010.data_ptr()), c_void_p(buf1011.data_ptr()), c_void_p(buf1014.data_ptr()), c_void_p(buf1015.data_ptr()), c_void_p(buf1016.data_ptr()), c_void_p(buf1018.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1020.data_ptr()), c_void_p(buf1022.data_ptr()), c_void_p(buf1023.data_ptr()), c_void_p(buf1024.data_ptr()), c_void_p(buf1026.data_ptr()), c_void_p(buf1027.data_ptr()), c_void_p(buf1028.data_ptr()), c_void_p(buf1030.data_ptr()), c_void_p(buf1031.data_ptr()), c_void_p(buf1032.data_ptr()), c_void_p(buf1034.data_ptr()), c_void_p(buf1035.data_ptr()), c_void_p(buf1036.data_ptr()), c_void_p(buf1038.data_ptr()), c_void_p(buf1039.data_ptr()), c_void_p(buf1040.data_ptr()), c_void_p(buf1042.data_ptr()), c_void_p(buf1043.data_ptr()), c_void_p(buf1044.data_ptr()), c_void_p(buf1046.data_ptr()), c_void_p(buf1047.data_ptr()), c_void_p(buf1048.data_ptr()), c_void_p(buf1050.data_ptr()), c_void_p(buf1051.data_ptr()), c_void_p(buf1052.data_ptr()), c_void_p(buf1054.data_ptr()), c_void_p(buf1055.data_ptr()), c_void_p(buf1056.data_ptr()), c_void_p(buf1058.data_ptr()), c_void_p(buf1059.data_ptr()), c_void_p(buf1060.data_ptr()), c_void_p(buf1062.data_ptr()), c_void_p(buf1063.data_ptr()), c_void_p(buf1064.data_ptr()), c_void_p(buf1066.data_ptr()), c_void_p(buf1067.data_ptr()), c_void_p(buf1068.data_ptr()), c_void_p(buf1070.data_ptr()), c_void_p(buf1071.data_ptr()), c_void_p(buf1072.data_ptr()), c_void_p(buf1074.data_ptr()), c_void_p(buf1075.data_ptr()), c_void_p(buf1076.data_ptr()), c_void_p(buf1078.data_ptr()), c_void_p(buf1079.data_ptr()), c_void_p(buf1080.data_ptr()), c_void_p(buf1082.data_ptr()), c_void_p(buf1083.data_ptr()), c_void_p(buf1084.data_ptr()), c_void_p(buf1086.data_ptr()), c_void_p(buf1087.data_ptr()), c_void_p(buf1088.data_ptr()), c_void_p(buf1090.data_ptr()), c_void_p(buf1091.data_ptr()), c_void_p(buf1092.data_ptr()), c_void_p(buf1094.data_ptr()), c_void_p(buf1095.data_ptr()), c_void_p(buf1096.data_ptr()), c_void_p(buf1098.data_ptr()), c_void_p(buf1099.data_ptr()), c_void_p(buf1100.data_ptr()), c_void_p(buf1102.data_ptr()), c_void_p(buf1103.data_ptr()), c_void_p(buf1104.data_ptr()), c_void_p(buf1106.data_ptr()), c_void_p(buf995.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(buf958.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf876.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf835.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf996.data_ptr()), c_void_p(buf998.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf1002.data_ptr()), c_void_p(buf999.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1003.data_ptr()), c_void_p(buf1004.data_ptr()), c_void_p(buf1001.data_ptr()), c_void_p(buf1005.data_ptr()), c_void_p(buf1107.data_ptr()), c_void_p(buf1006.data_ptr()), c_void_p(buf1007.data_ptr()), c_void_p(buf1008.data_ptr()), c_void_p(buf1009.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf1017.data_ptr()), c_void_p(buf1021.data_ptr()), c_void_p(buf1025.data_ptr()), c_void_p(buf1029.data_ptr()), c_void_p(buf1033.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(buf1041.data_ptr()), c_void_p(buf1045.data_ptr()), c_void_p(buf1049.data_ptr()), c_void_p(buf1053.data_ptr()), c_void_p(buf1057.data_ptr()), c_void_p(buf1061.data_ptr()), c_void_p(buf1065.data_ptr()), c_void_p(buf1069.data_ptr()), c_void_p(buf1073.data_ptr()), c_void_p(buf1077.data_ptr()), c_void_p(buf1081.data_ptr()), c_void_p(buf1085.data_ptr()), c_void_p(buf1089.data_ptr()), c_void_p(buf1093.data_ptr()), c_void_p(buf1097.data_ptr()), c_void_p(buf1101.data_ptr()), c_void_p(buf1105.data_ptr()))
    del buf1002
    del buf1003
    del buf101
    del buf138
    del buf14
    del buf142
    del buf179
    del buf18
    del buf183
    del buf220
    del buf224
    del buf261
    del buf265
    del buf302
    del buf306
    del buf343
    del buf347
    del buf384
    del buf388
    del buf425
    del buf429
    del buf466
    del buf470
    del buf507
    del buf511
    del buf548
    del buf552
    del buf56
    del buf589
    del buf593
    del buf60
    del buf630
    del buf634
    del buf671
    del buf675
    del buf712
    del buf716
    del buf753
    del buf757
    del buf794
    del buf798
    del buf835
    del buf839
    del buf876
    del buf880
    del buf917
    del buf921
    del buf958
    del buf97
    del buf995
    del buf998
    del buf999
    del primals_393
    del primals_394
    return (buf1107, buf996, buf997, primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_109, primals_115, primals_125, primals_131, primals_141, primals_147, primals_157, primals_163, primals_173, primals_179, primals_189, primals_195, primals_205, primals_211, primals_221, primals_227, primals_237, primals_243, primals_253, primals_259, primals_269, primals_275, primals_285, primals_291, primals_301, primals_307, primals_317, primals_323, primals_333, primals_339, primals_349, primals_355, primals_365, primals_371, primals_381, primals_387, primals_392, primals_391, buf3, buf8, buf9, buf23, buf27, buf32, buf37, buf38, buf39, buf40, buf45, buf50, buf51, buf64, buf68, buf73, buf78, buf79, buf80, buf81, buf86, buf91, buf92, buf105, buf109, buf114, buf119, buf120, buf121, buf122, buf127, buf132, buf133, buf146, buf150, buf155, buf160, buf161, buf162, buf163, buf168, buf173, buf174, buf187, buf191, buf196, buf201, buf202, buf203, buf204, buf209, buf214, buf215, buf228, buf232, buf237, buf242, buf243, buf244, buf245, buf250, buf255, buf256, buf269, buf273, buf278, buf283, buf284, buf285, buf286, buf291, buf296, buf297, buf310, buf314, buf319, buf324, buf325, buf326, buf327, buf332, buf337, buf338, buf351, buf355, buf360, buf365, buf366, buf367, buf368, buf373, buf378, buf379, buf392, buf396, buf401, buf406, buf407, buf408, buf409, buf414, buf419, buf420, buf433, buf437, buf442, buf447, buf448, buf449, buf450, buf455, buf460, buf461, buf474, buf478, buf483, buf488, buf489, buf490, buf491, buf496, buf501, buf502, buf515, buf519, buf524, buf529, buf530, buf531, buf532, buf537, buf542, buf543, buf556, buf560, buf565, buf570, buf571, buf572, buf573, buf578, buf583, buf584, buf597, buf601, buf606, buf611, buf612, buf613, buf614, buf619, buf624, buf625, buf638, buf642, buf647, buf652, buf653, buf654, buf655, buf660, buf665, buf666, buf679, buf683, buf688, buf693, buf694, buf695, buf696, buf701, buf706, buf707, buf720, buf724, buf729, buf734, buf735, buf736, buf737, buf742, buf747, buf748, buf761, buf765, buf770, buf775, buf776, buf777, buf778, buf783, buf788, buf789, buf802, buf806, buf811, buf816, buf817, buf818, buf819, buf824, buf829, buf830, buf843, buf847, buf852, buf857, buf858, buf859, buf860, buf865, buf870, buf871, buf884, buf888, buf893, buf898, buf899, buf900, buf901, buf906, buf911, buf912, buf925, buf929, buf934, buf939, buf940, buf941, buf942, buf947, buf952, buf953, buf966, buf970, buf975, buf980, buf981, buf982, buf983, buf988, buf993, buf994, buf1000, buf1001, buf1004, buf1005, buf1006, buf1007, buf1008, buf1009, reinterpret_tensor(primals_389, (2, 1536), (1536, 1), 0), buf1010, reinterpret_tensor(primals_385, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_383, (6144, 1536), (1536, 1), 0), buf1011, reinterpret_tensor(primals_379, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf967, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf968, (24, 64, 512), (32768, 1, 64), 0), buf1012, reinterpret_tensor(buf957, (24, 64, 512), (32768, 1, 64), 0), buf1013, reinterpret_tensor(primals_377, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_375, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_373, (1536, 1536), (1536, 1), 0), buf1014, reinterpret_tensor(primals_369, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_367, (6144, 1536), (1536, 1), 0), buf1015, reinterpret_tensor(primals_363, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf926, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf927, (24, 64, 512), (32768, 1, 64), 0), buf1016, reinterpret_tensor(buf916, (24, 64, 512), (32768, 1, 64), 0), buf1017, reinterpret_tensor(primals_361, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_359, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_357, (1536, 1536), (1536, 1), 0), buf1018, reinterpret_tensor(primals_353, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_351, (6144, 1536), (1536, 1), 0), buf1019, reinterpret_tensor(primals_347, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf885, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf886, (24, 64, 512), (32768, 1, 64), 0), buf1020, reinterpret_tensor(buf875, (24, 64, 512), (32768, 1, 64), 0), buf1021, reinterpret_tensor(primals_345, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_343, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_341, (1536, 1536), (1536, 1), 0), buf1022, reinterpret_tensor(primals_337, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_335, (6144, 1536), (1536, 1), 0), buf1023, reinterpret_tensor(primals_331, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf844, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf845, (24, 64, 512), (32768, 1, 64), 0), buf1024, reinterpret_tensor(buf834, (24, 64, 512), (32768, 1, 64), 0), buf1025, reinterpret_tensor(primals_329, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_327, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_325, (1536, 1536), (1536, 1), 0), buf1026, reinterpret_tensor(primals_321, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_319, (6144, 1536), (1536, 1), 0), buf1027, reinterpret_tensor(primals_315, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf803, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf804, (24, 64, 512), (32768, 1, 64), 0), buf1028, reinterpret_tensor(buf793, (24, 64, 512), (32768, 1, 64), 0), buf1029, reinterpret_tensor(primals_313, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_311, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_309, (1536, 1536), (1536, 1), 0), buf1030, reinterpret_tensor(primals_305, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_303, (6144, 1536), (1536, 1), 0), buf1031, reinterpret_tensor(primals_299, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf762, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf763, (24, 64, 512), (32768, 1, 64), 0), buf1032, reinterpret_tensor(buf752, (24, 64, 512), (32768, 1, 64), 0), buf1033, reinterpret_tensor(primals_297, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_295, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_293, (1536, 1536), (1536, 1), 0), buf1034, reinterpret_tensor(primals_289, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_287, (6144, 1536), (1536, 1), 0), buf1035, reinterpret_tensor(primals_283, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf721, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf722, (24, 64, 512), (32768, 1, 64), 0), buf1036, reinterpret_tensor(buf711, (24, 64, 512), (32768, 1, 64), 0), buf1037, reinterpret_tensor(primals_281, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_279, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_277, (1536, 1536), (1536, 1), 0), buf1038, reinterpret_tensor(primals_273, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_271, (6144, 1536), (1536, 1), 0), buf1039, reinterpret_tensor(primals_267, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf680, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf681, (24, 64, 512), (32768, 1, 64), 0), buf1040, reinterpret_tensor(buf670, (24, 64, 512), (32768, 1, 64), 0), buf1041, reinterpret_tensor(primals_265, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_263, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_261, (1536, 1536), (1536, 1), 0), buf1042, reinterpret_tensor(primals_257, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_255, (6144, 1536), (1536, 1), 0), buf1043, reinterpret_tensor(primals_251, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf639, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf640, (24, 64, 512), (32768, 1, 64), 0), buf1044, reinterpret_tensor(buf629, (24, 64, 512), (32768, 1, 64), 0), buf1045, reinterpret_tensor(primals_249, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_247, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_245, (1536, 1536), (1536, 1), 0), buf1046, reinterpret_tensor(primals_241, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_239, (6144, 1536), (1536, 1), 0), buf1047, reinterpret_tensor(primals_235, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf598, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf599, (24, 64, 512), (32768, 1, 64), 0), buf1048, reinterpret_tensor(buf588, (24, 64, 512), (32768, 1, 64), 0), buf1049, reinterpret_tensor(primals_233, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_231, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_229, (1536, 1536), (1536, 1), 0), buf1050, reinterpret_tensor(primals_225, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_223, (6144, 1536), (1536, 1), 0), buf1051, reinterpret_tensor(primals_219, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf557, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf558, (24, 64, 512), (32768, 1, 64), 0), buf1052, reinterpret_tensor(buf547, (24, 64, 512), (32768, 1, 64), 0), buf1053, reinterpret_tensor(primals_217, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_215, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_213, (1536, 1536), (1536, 1), 0), buf1054, reinterpret_tensor(primals_209, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_207, (6144, 1536), (1536, 1), 0), buf1055, reinterpret_tensor(primals_203, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf516, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf517, (24, 64, 512), (32768, 1, 64), 0), buf1056, reinterpret_tensor(buf506, (24, 64, 512), (32768, 1, 64), 0), buf1057, reinterpret_tensor(primals_201, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_199, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_197, (1536, 1536), (1536, 1), 0), buf1058, reinterpret_tensor(primals_193, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_191, (6144, 1536), (1536, 1), 0), buf1059, reinterpret_tensor(primals_187, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf475, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf476, (24, 64, 512), (32768, 1, 64), 0), buf1060, reinterpret_tensor(buf465, (24, 64, 512), (32768, 1, 64), 0), buf1061, reinterpret_tensor(primals_185, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_183, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_181, (1536, 1536), (1536, 1), 0), buf1062, reinterpret_tensor(primals_177, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_175, (6144, 1536), (1536, 1), 0), buf1063, reinterpret_tensor(primals_171, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf434, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf435, (24, 64, 512), (32768, 1, 64), 0), buf1064, reinterpret_tensor(buf424, (24, 64, 512), (32768, 1, 64), 0), buf1065, reinterpret_tensor(primals_169, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_167, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_165, (1536, 1536), (1536, 1), 0), buf1066, reinterpret_tensor(primals_161, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_159, (6144, 1536), (1536, 1), 0), buf1067, reinterpret_tensor(primals_155, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf393, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf394, (24, 64, 512), (32768, 1, 64), 0), buf1068, reinterpret_tensor(buf383, (24, 64, 512), (32768, 1, 64), 0), buf1069, reinterpret_tensor(primals_153, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_151, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_149, (1536, 1536), (1536, 1), 0), buf1070, reinterpret_tensor(primals_145, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_143, (6144, 1536), (1536, 1), 0), buf1071, reinterpret_tensor(primals_139, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf352, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf353, (24, 64, 512), (32768, 1, 64), 0), buf1072, reinterpret_tensor(buf342, (24, 64, 512), (32768, 1, 64), 0), buf1073, reinterpret_tensor(primals_137, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_135, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_133, (1536, 1536), (1536, 1), 0), buf1074, reinterpret_tensor(primals_129, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_127, (6144, 1536), (1536, 1), 0), buf1075, reinterpret_tensor(primals_123, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf311, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf312, (24, 64, 512), (32768, 1, 64), 0), buf1076, reinterpret_tensor(buf301, (24, 64, 512), (32768, 1, 64), 0), buf1077, reinterpret_tensor(primals_121, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_119, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_117, (1536, 1536), (1536, 1), 0), buf1078, reinterpret_tensor(primals_113, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_111, (6144, 1536), (1536, 1), 0), buf1079, reinterpret_tensor(primals_107, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf270, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf271, (24, 64, 512), (32768, 1, 64), 0), buf1080, reinterpret_tensor(buf260, (24, 64, 512), (32768, 1, 64), 0), buf1081, reinterpret_tensor(primals_105, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_103, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_101, (1536, 1536), (1536, 1), 0), buf1082, reinterpret_tensor(primals_97, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_95, (6144, 1536), (1536, 1), 0), buf1083, reinterpret_tensor(primals_91, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf229, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf230, (24, 64, 512), (32768, 1, 64), 0), buf1084, reinterpret_tensor(buf219, (24, 64, 512), (32768, 1, 64), 0), buf1085, reinterpret_tensor(primals_89, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_87, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_85, (1536, 1536), (1536, 1), 0), buf1086, reinterpret_tensor(primals_81, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_79, (6144, 1536), (1536, 1), 0), buf1087, reinterpret_tensor(primals_75, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf188, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf189, (24, 64, 512), (32768, 1, 64), 0), buf1088, reinterpret_tensor(buf178, (24, 64, 512), (32768, 1, 64), 0), buf1089, reinterpret_tensor(primals_73, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_71, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_69, (1536, 1536), (1536, 1), 0), buf1090, reinterpret_tensor(primals_65, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_63, (6144, 1536), (1536, 1), 0), buf1091, reinterpret_tensor(primals_59, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf147, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf148, (24, 64, 512), (32768, 1, 64), 0), buf1092, reinterpret_tensor(buf137, (24, 64, 512), (32768, 1, 64), 0), buf1093, reinterpret_tensor(primals_57, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_55, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_53, (1536, 1536), (1536, 1), 0), buf1094, reinterpret_tensor(primals_49, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_47, (6144, 1536), (1536, 1), 0), buf1095, reinterpret_tensor(primals_43, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf106, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf107, (24, 64, 512), (32768, 1, 64), 0), buf1096, reinterpret_tensor(buf96, (24, 64, 512), (32768, 1, 64), 0), buf1097, reinterpret_tensor(primals_41, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_39, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_37, (1536, 1536), (1536, 1), 0), buf1098, reinterpret_tensor(primals_33, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_31, (6144, 1536), (1536, 1), 0), buf1099, reinterpret_tensor(primals_27, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf65, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf66, (24, 64, 512), (32768, 1, 64), 0), buf1100, reinterpret_tensor(buf55, (24, 64, 512), (32768, 1, 64), 0), buf1101, reinterpret_tensor(primals_25, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_23, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_21, (1536, 1536), (1536, 1), 0), buf1102, reinterpret_tensor(primals_17, (1536, 6144), (6144, 1), 0), reinterpret_tensor(primals_15, (6144, 1536), (1536, 1), 0), buf1103, reinterpret_tensor(primals_11, (1536, 1536), (1536, 1), 0), reinterpret_tensor(buf24, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf25, (24, 64, 512), (32768, 1, 64), 0), buf1104, reinterpret_tensor(buf13, (24, 64, 512), (32768, 1, 64), 0), buf1105, reinterpret_tensor(primals_9, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_7, (1536, 1536), (1536, 1), 0), reinterpret_tensor(primals_5, (1536, 1536), (1536, 1), 0), buf1106, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128100, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((1536, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((6144, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((1536, 6144), (6144, 1), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((2, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_392 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_393 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    primals_394 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaV2ForQuestionAnswering', benchmark_compiled_module)
