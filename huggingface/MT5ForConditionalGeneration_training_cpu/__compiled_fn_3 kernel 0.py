
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
                out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp22;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr1[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr1[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                        auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp28);
                    }
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp27 = in_ptr1[static_cast<long>(x0 + (6L*tmp26))];
                    auto tmp28 = decltype(tmp0)(tmp0 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = std::exp(tmp30);
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_embedding_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
                out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp16;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr1[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr1[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
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
                        auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                    auto tmp32 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
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
                    auto tmp21 = in_ptr1[static_cast<long>(x0 + (6L*tmp20))];
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
                    in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = tmp0.exp();
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp19.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__log_softmax_nll_loss_forward_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(250112L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (250112L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(250112L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (250112L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(250112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (250112L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (250112L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 250112);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 250112L), "index out of bounds: 0 <= tmp8 < 250112L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (250112L*x0))];
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193 = args
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
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (250112, 512), (512, 1))
    assert_size_stride(primals_44, (384, 512), (512, 1))
    assert_size_stride(primals_45, (384, 512), (512, 1))
    assert_size_stride(primals_46, (384, 512), (512, 1))
    assert_size_stride(primals_47, (32, 6), (6, 1))
    assert_size_stride(primals_48, (512, 384), (384, 1))
    assert_size_stride(primals_49, (1024, 512), (512, 1))
    assert_size_stride(primals_50, (1024, 512), (512, 1))
    assert_size_stride(primals_51, (512, 1024), (1024, 1))
    assert_size_stride(primals_52, (384, 512), (512, 1))
    assert_size_stride(primals_53, (384, 512), (512, 1))
    assert_size_stride(primals_54, (384, 512), (512, 1))
    assert_size_stride(primals_55, (512, 384), (384, 1))
    assert_size_stride(primals_56, (1024, 512), (512, 1))
    assert_size_stride(primals_57, (1024, 512), (512, 1))
    assert_size_stride(primals_58, (512, 1024), (1024, 1))
    assert_size_stride(primals_59, (384, 512), (512, 1))
    assert_size_stride(primals_60, (384, 512), (512, 1))
    assert_size_stride(primals_61, (384, 512), (512, 1))
    assert_size_stride(primals_62, (512, 384), (384, 1))
    assert_size_stride(primals_63, (1024, 512), (512, 1))
    assert_size_stride(primals_64, (1024, 512), (512, 1))
    assert_size_stride(primals_65, (512, 1024), (1024, 1))
    assert_size_stride(primals_66, (384, 512), (512, 1))
    assert_size_stride(primals_67, (384, 512), (512, 1))
    assert_size_stride(primals_68, (384, 512), (512, 1))
    assert_size_stride(primals_69, (512, 384), (384, 1))
    assert_size_stride(primals_70, (1024, 512), (512, 1))
    assert_size_stride(primals_71, (1024, 512), (512, 1))
    assert_size_stride(primals_72, (512, 1024), (1024, 1))
    assert_size_stride(primals_73, (384, 512), (512, 1))
    assert_size_stride(primals_74, (384, 512), (512, 1))
    assert_size_stride(primals_75, (384, 512), (512, 1))
    assert_size_stride(primals_76, (512, 384), (384, 1))
    assert_size_stride(primals_77, (1024, 512), (512, 1))
    assert_size_stride(primals_78, (1024, 512), (512, 1))
    assert_size_stride(primals_79, (512, 1024), (1024, 1))
    assert_size_stride(primals_80, (384, 512), (512, 1))
    assert_size_stride(primals_81, (384, 512), (512, 1))
    assert_size_stride(primals_82, (384, 512), (512, 1))
    assert_size_stride(primals_83, (512, 384), (384, 1))
    assert_size_stride(primals_84, (1024, 512), (512, 1))
    assert_size_stride(primals_85, (1024, 512), (512, 1))
    assert_size_stride(primals_86, (512, 1024), (1024, 1))
    assert_size_stride(primals_87, (384, 512), (512, 1))
    assert_size_stride(primals_88, (384, 512), (512, 1))
    assert_size_stride(primals_89, (384, 512), (512, 1))
    assert_size_stride(primals_90, (512, 384), (384, 1))
    assert_size_stride(primals_91, (1024, 512), (512, 1))
    assert_size_stride(primals_92, (1024, 512), (512, 1))
    assert_size_stride(primals_93, (512, 1024), (1024, 1))
    assert_size_stride(primals_94, (384, 512), (512, 1))
    assert_size_stride(primals_95, (384, 512), (512, 1))
    assert_size_stride(primals_96, (384, 512), (512, 1))
    assert_size_stride(primals_97, (512, 384), (384, 1))
    assert_size_stride(primals_98, (1024, 512), (512, 1))
    assert_size_stride(primals_99, (1024, 512), (512, 1))
    assert_size_stride(primals_100, (512, 1024), (1024, 1))
    assert_size_stride(primals_101, (384, 512), (512, 1))
    assert_size_stride(primals_102, (384, 512), (512, 1))
    assert_size_stride(primals_103, (384, 512), (512, 1))
    assert_size_stride(primals_104, (32, 6), (6, 1))
    assert_size_stride(primals_105, (512, 384), (384, 1))
    assert_size_stride(primals_106, (384, 512), (512, 1))
    assert_size_stride(primals_107, (384, 512), (512, 1))
    assert_size_stride(primals_108, (384, 512), (512, 1))
    assert_size_stride(primals_109, (512, 384), (384, 1))
    assert_size_stride(primals_110, (1024, 512), (512, 1))
    assert_size_stride(primals_111, (1024, 512), (512, 1))
    assert_size_stride(primals_112, (512, 1024), (1024, 1))
    assert_size_stride(primals_113, (384, 512), (512, 1))
    assert_size_stride(primals_114, (384, 512), (512, 1))
    assert_size_stride(primals_115, (384, 512), (512, 1))
    assert_size_stride(primals_116, (512, 384), (384, 1))
    assert_size_stride(primals_117, (384, 512), (512, 1))
    assert_size_stride(primals_118, (384, 512), (512, 1))
    assert_size_stride(primals_119, (384, 512), (512, 1))
    assert_size_stride(primals_120, (512, 384), (384, 1))
    assert_size_stride(primals_121, (1024, 512), (512, 1))
    assert_size_stride(primals_122, (1024, 512), (512, 1))
    assert_size_stride(primals_123, (512, 1024), (1024, 1))
    assert_size_stride(primals_124, (384, 512), (512, 1))
    assert_size_stride(primals_125, (384, 512), (512, 1))
    assert_size_stride(primals_126, (384, 512), (512, 1))
    assert_size_stride(primals_127, (512, 384), (384, 1))
    assert_size_stride(primals_128, (384, 512), (512, 1))
    assert_size_stride(primals_129, (384, 512), (512, 1))
    assert_size_stride(primals_130, (384, 512), (512, 1))
    assert_size_stride(primals_131, (512, 384), (384, 1))
    assert_size_stride(primals_132, (1024, 512), (512, 1))
    assert_size_stride(primals_133, (1024, 512), (512, 1))
    assert_size_stride(primals_134, (512, 1024), (1024, 1))
    assert_size_stride(primals_135, (384, 512), (512, 1))
    assert_size_stride(primals_136, (384, 512), (512, 1))
    assert_size_stride(primals_137, (384, 512), (512, 1))
    assert_size_stride(primals_138, (512, 384), (384, 1))
    assert_size_stride(primals_139, (384, 512), (512, 1))
    assert_size_stride(primals_140, (384, 512), (512, 1))
    assert_size_stride(primals_141, (384, 512), (512, 1))
    assert_size_stride(primals_142, (512, 384), (384, 1))
    assert_size_stride(primals_143, (1024, 512), (512, 1))
    assert_size_stride(primals_144, (1024, 512), (512, 1))
    assert_size_stride(primals_145, (512, 1024), (1024, 1))
    assert_size_stride(primals_146, (384, 512), (512, 1))
    assert_size_stride(primals_147, (384, 512), (512, 1))
    assert_size_stride(primals_148, (384, 512), (512, 1))
    assert_size_stride(primals_149, (512, 384), (384, 1))
    assert_size_stride(primals_150, (384, 512), (512, 1))
    assert_size_stride(primals_151, (384, 512), (512, 1))
    assert_size_stride(primals_152, (384, 512), (512, 1))
    assert_size_stride(primals_153, (512, 384), (384, 1))
    assert_size_stride(primals_154, (1024, 512), (512, 1))
    assert_size_stride(primals_155, (1024, 512), (512, 1))
    assert_size_stride(primals_156, (512, 1024), (1024, 1))
    assert_size_stride(primals_157, (384, 512), (512, 1))
    assert_size_stride(primals_158, (384, 512), (512, 1))
    assert_size_stride(primals_159, (384, 512), (512, 1))
    assert_size_stride(primals_160, (512, 384), (384, 1))
    assert_size_stride(primals_161, (384, 512), (512, 1))
    assert_size_stride(primals_162, (384, 512), (512, 1))
    assert_size_stride(primals_163, (384, 512), (512, 1))
    assert_size_stride(primals_164, (512, 384), (384, 1))
    assert_size_stride(primals_165, (1024, 512), (512, 1))
    assert_size_stride(primals_166, (1024, 512), (512, 1))
    assert_size_stride(primals_167, (512, 1024), (1024, 1))
    assert_size_stride(primals_168, (384, 512), (512, 1))
    assert_size_stride(primals_169, (384, 512), (512, 1))
    assert_size_stride(primals_170, (384, 512), (512, 1))
    assert_size_stride(primals_171, (512, 384), (384, 1))
    assert_size_stride(primals_172, (384, 512), (512, 1))
    assert_size_stride(primals_173, (384, 512), (512, 1))
    assert_size_stride(primals_174, (384, 512), (512, 1))
    assert_size_stride(primals_175, (512, 384), (384, 1))
    assert_size_stride(primals_176, (1024, 512), (512, 1))
    assert_size_stride(primals_177, (1024, 512), (512, 1))
    assert_size_stride(primals_178, (512, 1024), (1024, 1))
    assert_size_stride(primals_179, (384, 512), (512, 1))
    assert_size_stride(primals_180, (384, 512), (512, 1))
    assert_size_stride(primals_181, (384, 512), (512, 1))
    assert_size_stride(primals_182, (512, 384), (384, 1))
    assert_size_stride(primals_183, (384, 512), (512, 1))
    assert_size_stride(primals_184, (384, 512), (512, 1))
    assert_size_stride(primals_185, (384, 512), (512, 1))
    assert_size_stride(primals_186, (512, 384), (384, 1))
    assert_size_stride(primals_187, (1024, 512), (512, 1))
    assert_size_stride(primals_188, (1024, 512), (512, 1))
    assert_size_stride(primals_189, (512, 1024), (1024, 1))
    assert_size_stride(primals_190, (250112, 512), (512, 1))
    assert_size_stride(primals_191, (1, 128), (128, 1))
    assert_size_stride(primals_192, (1, 128), (128, 1))
    assert_size_stride(primals_193, (1, 128), (128, 1))
    buf0 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_0(c_void_p(primals_191.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [hidden_states, inputs_embeds], Original ATen: [aten.embedding, aten.native_dropout]
    buf1 = aten.native_dropout(buf0, 0.1, True)
    buf2 = buf1[0]
    buf3 = buf1[1]
    del buf1
    buf4 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 1), 0); del buf4  # reuse
    buf6 = reinterpret_tensor(buf0, (128, 512), (512, 1), 0); del buf0  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_1(c_void_p(buf5.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf6.data_ptr()))
    buf7 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_44, (512, 384), (1, 512), 0), out=buf7)
    buf8 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_45, (512, 384), (1, 512), 0), out=buf8)
    buf9 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf6, reinterpret_tensor(primals_46, (512, 384), (1, 512), 0), out=buf9)
    buf10 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf8, (6, 64, 128), (64, 1, 384), 0), out=buf10)
    buf11 = empty((128, 128), device='cpu', dtype=torch.int64)
    buf12 = empty_strided((1, 6, 128, 1), (768, 128, 1, 768), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf10, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf10  # reuse
    buf14 = empty_strided((1, 6, 128, 1), (768, 128, 1, 768), device='cpu', dtype=torch.float32)
    buf15 = buf13; del buf13  # reuse
    cpp_fused__softmax__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_2(c_void_p(buf15.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    # Source Nodes: [attn_weights_1, softmax], Original ATen: [aten._softmax, aten.native_dropout]
    buf16 = aten.native_dropout(buf15, 0.1, True)
    buf17 = buf16[0]
    buf18 = buf16[1]
    del buf16
    buf19 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf17, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf9, (6, 128, 64), (64, 384, 1), 0), out=buf19)
    buf20 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_3(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, reinterpret_tensor(primals_48, (384, 512), (1, 384), 0), out=buf21)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf22 = aten.native_dropout(reinterpret_tensor(buf21, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf25 = buf23; del buf23  # reuse
    buf26 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf26, (1, 128, 1), (128, 1, 1), 0); del buf26  # reuse
    buf28 = buf21; del buf21  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_4(c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf28, reinterpret_tensor(primals_49, (512, 1024), (1, 512), 0), out=buf29)
    buf31 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear], Original ATen: [aten.mm]
    extern_kernels.mm(buf28, reinterpret_tensor(primals_50, (512, 1024), (1, 512), 0), out=buf31)
    buf30 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_5(c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    # Source Nodes: [add_7, hidden_gelu, hidden_states_7, hidden_states_8, mul_7], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf33 = aten.native_dropout(buf32, 0.1, True)
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_51, (1024, 512), (1, 1024), 0), out=buf36)
    # Source Nodes: [l__mod___encoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf37 = aten.native_dropout(reinterpret_tensor(buf36, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf38 = buf37[0]
    buf39 = buf37[1]
    del buf37
    buf40 = buf38; del buf38  # reuse
    buf41 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf41, (1, 128, 1), (128, 1, 1), 0); del buf41  # reuse
    buf43 = buf36; del buf36  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_6(c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = reinterpret_tensor(buf19, (128, 384), (384, 1), 0); del buf19  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, reinterpret_tensor(primals_52, (512, 384), (1, 512), 0), out=buf44)
    buf45 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, reinterpret_tensor(primals_53, (512, 384), (1, 512), 0), out=buf45)
    buf46 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, reinterpret_tensor(primals_54, (512, 384), (1, 512), 0), out=buf46)
    buf47 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf45, (6, 64, 128), (64, 1, 384), 0), out=buf47)
    buf48 = buf14; del buf14  # reuse
    buf49 = reinterpret_tensor(buf47, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf47  # reuse
    buf50 = buf12; del buf12  # reuse
    buf51 = buf49; del buf49  # reuse
    cpp_fused__softmax_7(c_void_p(buf51.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    # Source Nodes: [attn_weights_3, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
    buf52 = aten.native_dropout(buf51, 0.1, True)
    buf53 = buf52[0]
    buf54 = buf52[1]
    del buf52
    buf55 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf46, (6, 128, 64), (64, 384, 1), 0), out=buf55)
    buf56 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_55, (384, 512), (1, 384), 0), out=buf57)
    # Source Nodes: [l__mod___encoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf58 = aten.native_dropout(reinterpret_tensor(buf57, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf59 = buf58[0]
    buf60 = buf58[1]
    del buf58
    buf61 = buf59; del buf59  # reuse
    buf62 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf63 = reinterpret_tensor(buf62, (1, 128, 1), (128, 1, 1), 0); del buf62  # reuse
    buf64 = buf57; del buf57  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_9(c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = reinterpret_tensor(buf32, (128, 1024), (1024, 1), 0); del buf32  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf64, reinterpret_tensor(primals_56, (512, 1024), (1, 512), 0), out=buf65)
    buf67 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf64, reinterpret_tensor(primals_57, (512, 1024), (1, 512), 0), out=buf67)
    buf66 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf68 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_10(c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()))
    # Source Nodes: [add_13, hidden_gelu_1, hidden_states_19, hidden_states_20, mul_16], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf69 = aten.native_dropout(buf68, 0.1, True)
    buf70 = buf69[0]
    buf71 = buf69[1]
    del buf69
    buf72 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_58, (1024, 512), (1, 1024), 0), out=buf72)
    # Source Nodes: [l__mod___encoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf73 = aten.native_dropout(reinterpret_tensor(buf72, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf74; del buf74  # reuse
    buf77 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf78 = reinterpret_tensor(buf77, (1, 128, 1), (128, 1, 1), 0); del buf77  # reuse
    buf79 = buf72; del buf72  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_11(c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf55, (128, 384), (384, 1), 0); del buf55  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf79, reinterpret_tensor(primals_59, (512, 384), (1, 512), 0), out=buf80)
    buf81 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf79, reinterpret_tensor(primals_60, (512, 384), (1, 512), 0), out=buf81)
    buf82 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf79, reinterpret_tensor(primals_61, (512, 384), (1, 512), 0), out=buf82)
    buf83 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf80, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf81, (6, 64, 128), (64, 1, 384), 0), out=buf83)
    buf84 = buf50; del buf50  # reuse
    buf85 = reinterpret_tensor(buf83, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf83  # reuse
    buf86 = buf48; del buf48  # reuse
    buf87 = buf85; del buf85  # reuse
    cpp_fused__softmax_12(c_void_p(buf87.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    # Source Nodes: [attn_weights_5, softmax_2], Original ATen: [aten._softmax, aten.native_dropout]
    buf88 = aten.native_dropout(buf87, 0.1, True)
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf89, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf82, (6, 128, 64), (64, 384, 1), 0), out=buf91)
    buf92 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_13(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf92, reinterpret_tensor(primals_62, (384, 512), (1, 384), 0), out=buf93)
    # Source Nodes: [l__mod___encoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf94 = aten.native_dropout(reinterpret_tensor(buf93, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = buf95; del buf95  # reuse
    buf98 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf99 = reinterpret_tensor(buf98, (1, 128, 1), (128, 1, 1), 0); del buf98  # reuse
    buf100 = buf93; del buf93  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_14(c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf68, (128, 1024), (1024, 1), 0); del buf68  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf100, reinterpret_tensor(primals_63, (512, 1024), (1, 512), 0), out=buf101)
    buf103 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf100, reinterpret_tensor(primals_64, (512, 1024), (1, 512), 0), out=buf103)
    buf102 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf104 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_15(c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()))
    # Source Nodes: [add_19, hidden_gelu_2, hidden_states_31, hidden_states_32, mul_25], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf105 = aten.native_dropout(buf104, 0.1, True)
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_65, (1024, 512), (1, 1024), 0), out=buf108)
    # Source Nodes: [l__mod___encoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf109 = aten.native_dropout(reinterpret_tensor(buf108, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf110 = buf109[0]
    buf111 = buf109[1]
    del buf109
    buf112 = buf110; del buf110  # reuse
    buf113 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf114 = reinterpret_tensor(buf113, (1, 128, 1), (128, 1, 1), 0); del buf113  # reuse
    buf115 = buf108; del buf108  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_16(c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf91, (128, 384), (384, 1), 0); del buf91  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf115, reinterpret_tensor(primals_66, (512, 384), (1, 512), 0), out=buf116)
    buf117 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf115, reinterpret_tensor(primals_67, (512, 384), (1, 512), 0), out=buf117)
    buf118 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf115, reinterpret_tensor(primals_68, (512, 384), (1, 512), 0), out=buf118)
    buf119 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf117, (6, 64, 128), (64, 1, 384), 0), out=buf119)
    buf120 = buf86; del buf86  # reuse
    buf121 = reinterpret_tensor(buf119, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf119  # reuse
    buf122 = buf84; del buf84  # reuse
    buf123 = buf121; del buf121  # reuse
    cpp_fused__softmax_17(c_void_p(buf123.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    # Source Nodes: [attn_weights_7, softmax_3], Original ATen: [aten._softmax, aten.native_dropout]
    buf124 = aten.native_dropout(buf123, 0.1, True)
    buf125 = buf124[0]
    buf126 = buf124[1]
    del buf124
    buf127 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf125, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf118, (6, 128, 64), (64, 384, 1), 0), out=buf127)
    buf128 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_18(c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf128, reinterpret_tensor(primals_69, (384, 512), (1, 384), 0), out=buf129)
    # Source Nodes: [l__mod___encoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf130 = aten.native_dropout(reinterpret_tensor(buf129, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf131 = buf130[0]
    buf132 = buf130[1]
    del buf130
    buf133 = buf131; del buf131  # reuse
    buf134 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf134, (1, 128, 1), (128, 1, 1), 0); del buf134  # reuse
    buf136 = buf129; del buf129  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_19(c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf136.data_ptr()))
    buf137 = reinterpret_tensor(buf104, (128, 1024), (1024, 1), 0); del buf104  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf136, reinterpret_tensor(primals_70, (512, 1024), (1, 512), 0), out=buf137)
    buf139 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf136, reinterpret_tensor(primals_71, (512, 1024), (1, 512), 0), out=buf139)
    buf138 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf140 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_20(c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [add_25, hidden_gelu_3, hidden_states_43, hidden_states_44, mul_34], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf141 = aten.native_dropout(buf140, 0.1, True)
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf144 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_72, (1024, 512), (1, 1024), 0), out=buf144)
    # Source Nodes: [l__mod___encoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf145 = aten.native_dropout(reinterpret_tensor(buf144, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf146 = buf145[0]
    buf147 = buf145[1]
    del buf145
    buf148 = buf146; del buf146  # reuse
    buf149 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf150 = reinterpret_tensor(buf149, (1, 128, 1), (128, 1, 1), 0); del buf149  # reuse
    buf151 = buf144; del buf144  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_21(c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = reinterpret_tensor(buf127, (128, 384), (384, 1), 0); del buf127  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf151, reinterpret_tensor(primals_73, (512, 384), (1, 512), 0), out=buf152)
    buf153 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf151, reinterpret_tensor(primals_74, (512, 384), (1, 512), 0), out=buf153)
    buf154 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf151, reinterpret_tensor(primals_75, (512, 384), (1, 512), 0), out=buf154)
    buf155 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf153, (6, 64, 128), (64, 1, 384), 0), out=buf155)
    buf156 = buf122; del buf122  # reuse
    buf157 = reinterpret_tensor(buf155, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf155  # reuse
    buf158 = buf120; del buf120  # reuse
    buf159 = buf157; del buf157  # reuse
    cpp_fused__softmax_22(c_void_p(buf159.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    # Source Nodes: [attn_weights_9, softmax_4], Original ATen: [aten._softmax, aten.native_dropout]
    buf160 = aten.native_dropout(buf159, 0.1, True)
    buf161 = buf160[0]
    buf162 = buf160[1]
    del buf160
    buf163 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf161, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf154, (6, 128, 64), (64, 384, 1), 0), out=buf163)
    buf164 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_23(c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf164, reinterpret_tensor(primals_76, (384, 512), (1, 384), 0), out=buf165)
    # Source Nodes: [l__mod___encoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf166 = aten.native_dropout(reinterpret_tensor(buf165, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf167 = buf166[0]
    buf168 = buf166[1]
    del buf166
    buf169 = buf167; del buf167  # reuse
    buf170 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf170, (1, 128, 1), (128, 1, 1), 0); del buf170  # reuse
    buf172 = buf165; del buf165  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_24(c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf140, (128, 1024), (1024, 1), 0); del buf140  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf172, reinterpret_tensor(primals_77, (512, 1024), (1, 512), 0), out=buf173)
    buf175 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf172, reinterpret_tensor(primals_78, (512, 1024), (1, 512), 0), out=buf175)
    buf174 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf176 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_25(c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()))
    # Source Nodes: [add_31, hidden_gelu_4, hidden_states_55, hidden_states_56, mul_43], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf177 = aten.native_dropout(buf176, 0.1, True)
    buf178 = buf177[0]
    buf179 = buf177[1]
    del buf177
    buf180 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_79, (1024, 512), (1, 1024), 0), out=buf180)
    # Source Nodes: [l__mod___encoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf181 = aten.native_dropout(reinterpret_tensor(buf180, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf182 = buf181[0]
    buf183 = buf181[1]
    del buf181
    buf184 = buf182; del buf182  # reuse
    buf185 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf186 = reinterpret_tensor(buf185, (1, 128, 1), (128, 1, 1), 0); del buf185  # reuse
    buf187 = buf180; del buf180  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_26(c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf163, (128, 384), (384, 1), 0); del buf163  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_80, (512, 384), (1, 512), 0), out=buf188)
    buf189 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_81, (512, 384), (1, 512), 0), out=buf189)
    buf190 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_82, (512, 384), (1, 512), 0), out=buf190)
    buf191 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf189, (6, 64, 128), (64, 1, 384), 0), out=buf191)
    buf192 = buf158; del buf158  # reuse
    buf193 = reinterpret_tensor(buf191, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf191  # reuse
    buf194 = buf156; del buf156  # reuse
    buf195 = buf193; del buf193  # reuse
    cpp_fused__softmax_27(c_void_p(buf195.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()))
    # Source Nodes: [attn_weights_11, softmax_5], Original ATen: [aten._softmax, aten.native_dropout]
    buf196 = aten.native_dropout(buf195, 0.1, True)
    buf197 = buf196[0]
    buf198 = buf196[1]
    del buf196
    buf199 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf197, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf190, (6, 128, 64), (64, 384, 1), 0), out=buf199)
    buf200 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_28(c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf200, reinterpret_tensor(primals_83, (384, 512), (1, 384), 0), out=buf201)
    # Source Nodes: [l__mod___encoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf202 = aten.native_dropout(reinterpret_tensor(buf201, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf203 = buf202[0]
    buf204 = buf202[1]
    del buf202
    buf205 = buf203; del buf203  # reuse
    buf206 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf207 = reinterpret_tensor(buf206, (1, 128, 1), (128, 1, 1), 0); del buf206  # reuse
    buf208 = buf201; del buf201  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_29(c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = reinterpret_tensor(buf176, (128, 1024), (1024, 1), 0); del buf176  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf208, reinterpret_tensor(primals_84, (512, 1024), (1, 512), 0), out=buf209)
    buf211 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf208, reinterpret_tensor(primals_85, (512, 1024), (1, 512), 0), out=buf211)
    buf210 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf212 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_30(c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    # Source Nodes: [add_37, hidden_gelu_5, hidden_states_67, hidden_states_68, mul_52], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf213 = aten.native_dropout(buf212, 0.1, True)
    buf214 = buf213[0]
    buf215 = buf213[1]
    del buf213
    buf216 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf214, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_86, (1024, 512), (1, 1024), 0), out=buf216)
    # Source Nodes: [l__mod___encoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf217 = aten.native_dropout(reinterpret_tensor(buf216, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf218 = buf217[0]
    buf219 = buf217[1]
    del buf217
    buf220 = buf218; del buf218  # reuse
    buf221 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf221, (1, 128, 1), (128, 1, 1), 0); del buf221  # reuse
    buf223 = buf216; del buf216  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_31(c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf199, (128, 384), (384, 1), 0); del buf199  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf223, reinterpret_tensor(primals_87, (512, 384), (1, 512), 0), out=buf224)
    buf225 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf223, reinterpret_tensor(primals_88, (512, 384), (1, 512), 0), out=buf225)
    buf226 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf223, reinterpret_tensor(primals_89, (512, 384), (1, 512), 0), out=buf226)
    buf227 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf224, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf225, (6, 64, 128), (64, 1, 384), 0), out=buf227)
    buf228 = buf194; del buf194  # reuse
    buf229 = reinterpret_tensor(buf227, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf227  # reuse
    buf230 = buf192; del buf192  # reuse
    buf231 = buf229; del buf229  # reuse
    cpp_fused__softmax_32(c_void_p(buf231.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    # Source Nodes: [attn_weights_13, softmax_6], Original ATen: [aten._softmax, aten.native_dropout]
    buf232 = aten.native_dropout(buf231, 0.1, True)
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf226, (6, 128, 64), (64, 384, 1), 0), out=buf235)
    buf236 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_33(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf236, reinterpret_tensor(primals_90, (384, 512), (1, 384), 0), out=buf237)
    # Source Nodes: [l__mod___encoder_block_6_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf238 = aten.native_dropout(reinterpret_tensor(buf237, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    buf241 = buf239; del buf239  # reuse
    buf242 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf242, (1, 128, 1), (128, 1, 1), 0); del buf242  # reuse
    buf244 = buf237; del buf237  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_34(c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf244.data_ptr()))
    buf245 = reinterpret_tensor(buf212, (128, 1024), (1024, 1), 0); del buf212  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf244, reinterpret_tensor(primals_91, (512, 1024), (1, 512), 0), out=buf245)
    buf247 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf244, reinterpret_tensor(primals_92, (512, 1024), (1, 512), 0), out=buf247)
    buf246 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf248 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_35(c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()))
    # Source Nodes: [add_43, hidden_gelu_6, hidden_states_79, hidden_states_80, mul_61], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf249 = aten.native_dropout(buf248, 0.1, True)
    buf250 = buf249[0]
    buf251 = buf249[1]
    del buf249
    buf252 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_93, (1024, 512), (1, 1024), 0), out=buf252)
    # Source Nodes: [l__mod___encoder_block_6_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf253 = aten.native_dropout(reinterpret_tensor(buf252, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = buf254; del buf254  # reuse
    buf257 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf258 = reinterpret_tensor(buf257, (1, 128, 1), (128, 1, 1), 0); del buf257  # reuse
    buf259 = buf252; del buf252  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_36(c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf259.data_ptr()))
    buf260 = reinterpret_tensor(buf235, (128, 384), (384, 1), 0); del buf235  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf259, reinterpret_tensor(primals_94, (512, 384), (1, 512), 0), out=buf260)
    buf261 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf259, reinterpret_tensor(primals_95, (512, 384), (1, 512), 0), out=buf261)
    buf262 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf259, reinterpret_tensor(primals_96, (512, 384), (1, 512), 0), out=buf262)
    buf263 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf260, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf261, (6, 64, 128), (64, 1, 384), 0), out=buf263)
    buf264 = buf230; del buf230  # reuse
    buf265 = reinterpret_tensor(buf263, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf263  # reuse
    buf266 = buf228; del buf228  # reuse
    buf267 = buf265; del buf265  # reuse
    cpp_fused__softmax_37(c_void_p(buf267.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del primals_47
    # Source Nodes: [attn_weights_15, softmax_7], Original ATen: [aten._softmax, aten.native_dropout]
    buf268 = aten.native_dropout(buf267, 0.1, True)
    buf269 = buf268[0]
    buf270 = buf268[1]
    del buf268
    buf271 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf269, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf262, (6, 128, 64), (64, 384, 1), 0), out=buf271)
    buf272 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_38(c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf272, reinterpret_tensor(primals_97, (384, 512), (1, 384), 0), out=buf273)
    # Source Nodes: [l__mod___encoder_block_7_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf274 = aten.native_dropout(reinterpret_tensor(buf273, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf277 = buf275; del buf275  # reuse
    buf278 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf279 = reinterpret_tensor(buf278, (1, 128, 1), (128, 1, 1), 0); del buf278  # reuse
    buf280 = buf273; del buf273  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_39(c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = reinterpret_tensor(buf248, (128, 1024), (1024, 1), 0); del buf248  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf280, reinterpret_tensor(primals_98, (512, 1024), (1, 512), 0), out=buf281)
    buf283 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_7], Original ATen: [aten.mm]
    extern_kernels.mm(buf280, reinterpret_tensor(primals_99, (512, 1024), (1, 512), 0), out=buf283)
    buf282 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf284 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_40(c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    # Source Nodes: [add_49, hidden_gelu_7, hidden_states_91, hidden_states_92, mul_70], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf285 = aten.native_dropout(buf284, 0.1, True)
    buf286 = buf285[0]
    buf287 = buf285[1]
    del buf285
    buf288 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_100, (1024, 512), (1, 1024), 0), out=buf288)
    # Source Nodes: [l__mod___encoder_block_7_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf289 = aten.native_dropout(reinterpret_tensor(buf288, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf290 = buf289[0]
    buf291 = buf289[1]
    del buf289
    buf292 = buf290; del buf290  # reuse
    buf293 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf294 = reinterpret_tensor(buf293, (1, 128, 1), (128, 1, 1), 0); del buf293  # reuse
    buf295 = reinterpret_tensor(buf288, (1, 128, 512), (65536, 512, 1), 0); del buf288  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_41(c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf295.data_ptr()))
    # Source Nodes: [hidden_states_100, hidden_states_97, hidden_states_98], Original ATen: [aten.mul, aten.native_dropout]
    buf296 = aten.native_dropout(buf295, 0.1, True)
    buf297 = buf296[0]
    buf298 = buf296[1]
    del buf296
    buf299 = buf295; del buf295  # reuse
    cpp_fused_embedding_42(c_void_p(primals_193.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf299.data_ptr()))
    del primals_43
    # Source Nodes: [hidden_states_101, inputs_embeds_1], Original ATen: [aten.embedding, aten.native_dropout]
    buf300 = aten.native_dropout(buf299, 0.1, True)
    buf301 = buf300[0]
    buf302 = buf300[1]
    del buf300
    buf303 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf304 = reinterpret_tensor(buf303, (1, 128, 1), (128, 1, 1), 0); del buf303  # reuse
    buf305 = reinterpret_tensor(buf299, (128, 512), (512, 1), 0); del buf299  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_43(c_void_p(buf304.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf271, (128, 384), (384, 1), 0); del buf271  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf305, reinterpret_tensor(primals_101, (512, 384), (1, 512), 0), out=buf306)
    buf307 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf305, reinterpret_tensor(primals_102, (512, 384), (1, 512), 0), out=buf307)
    buf308 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf305, reinterpret_tensor(primals_103, (512, 384), (1, 512), 0), out=buf308)
    buf309 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf306, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf307, (6, 64, 128), (64, 1, 384), 0), out=buf309)
    buf310 = empty((128, 128), device='cpu', dtype=torch.int64)
    buf311 = buf266; del buf266  # reuse
    buf312 = reinterpret_tensor(buf309, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf309  # reuse
    buf313 = buf312; del buf312  # reuse
    buf314 = buf264; del buf264  # reuse
    buf315 = buf313; del buf313  # reuse
    cpp_fused__softmax__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_44(c_void_p(buf315.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf314.data_ptr()))
    # Source Nodes: [attn_weights_17, softmax_8], Original ATen: [aten._softmax, aten.native_dropout]
    buf316 = aten.native_dropout(buf315, 0.1, True)
    buf317 = buf316[0]
    buf318 = buf316[1]
    del buf316
    buf319 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf317, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf308, (6, 128, 64), (64, 384, 1), 0), out=buf319)
    buf320 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_45(c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_105, (384, 512), (1, 384), 0), out=buf321)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf322 = aten.native_dropout(reinterpret_tensor(buf321, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf323 = buf322[0]
    buf324 = buf322[1]
    del buf322
    buf325 = buf323; del buf323  # reuse
    buf326 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf327 = reinterpret_tensor(buf326, (1, 128, 1), (128, 1, 1), 0); del buf326  # reuse
    buf328 = buf321; del buf321  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_46(c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = reinterpret_tensor(buf319, (128, 384), (384, 1), 0); del buf319  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf328, reinterpret_tensor(primals_106, (512, 384), (1, 512), 0), out=buf329)
    buf330 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_107, (512, 384), (1, 512), 0), out=buf330)
    buf331 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 384), (1, 512), 0), out=buf331)
    buf332 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf330, (6, 64, 128), (64, 1, 384), 0), out=buf332)
    buf333 = buf314; del buf314  # reuse
    buf334 = reinterpret_tensor(buf332, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf332  # reuse
    buf335 = buf311; del buf311  # reuse
    buf336 = buf334; del buf334  # reuse
    cpp_fused__softmax_47(c_void_p(buf336.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()))
    # Source Nodes: [attn_weights_19, softmax_9], Original ATen: [aten._softmax, aten.native_dropout]
    buf337 = aten.native_dropout(buf336, 0.1, True)
    buf338 = buf337[0]
    buf339 = buf337[1]
    del buf337
    buf340 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf338, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf331, (6, 128, 64), (64, 384, 1), 0), out=buf340)
    buf341 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_48(c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    buf342 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(buf341, reinterpret_tensor(primals_109, (384, 512), (1, 384), 0), out=buf342)
    # Source Nodes: [l__mod___decoder_block_0_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf343 = aten.native_dropout(reinterpret_tensor(buf342, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = buf344; del buf344  # reuse
    buf347 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf347, (1, 128, 1), (128, 1, 1), 0); del buf347  # reuse
    buf349 = buf342; del buf342  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_49(c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf349.data_ptr()))
    buf350 = reinterpret_tensor(buf284, (128, 1024), (1024, 1), 0); del buf284  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf349, reinterpret_tensor(primals_110, (512, 1024), (1, 512), 0), out=buf350)
    buf352 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf349, reinterpret_tensor(primals_111, (512, 1024), (1, 512), 0), out=buf352)
    buf351 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf353 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_50(c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()))
    # Source Nodes: [add_62, hidden_gelu_8, hidden_states_112, hidden_states_113, mul_87], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf354 = aten.native_dropout(buf353, 0.1, True)
    buf355 = buf354[0]
    buf356 = buf354[1]
    del buf354
    buf357 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_112, (1024, 512), (1, 1024), 0), out=buf357)
    # Source Nodes: [l__mod___decoder_block_0_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf358 = aten.native_dropout(reinterpret_tensor(buf357, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf359 = buf358[0]
    buf360 = buf358[1]
    del buf358
    buf361 = buf359; del buf359  # reuse
    buf362 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf362, (1, 128, 1), (128, 1, 1), 0); del buf362  # reuse
    buf364 = buf357; del buf357  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_51(c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = reinterpret_tensor(buf340, (128, 384), (384, 1), 0); del buf340  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf364, reinterpret_tensor(primals_113, (512, 384), (1, 512), 0), out=buf365)
    buf366 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf364, reinterpret_tensor(primals_114, (512, 384), (1, 512), 0), out=buf366)
    buf367 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf364, reinterpret_tensor(primals_115, (512, 384), (1, 512), 0), out=buf367)
    buf368 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf365, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf366, (6, 64, 128), (64, 1, 384), 0), out=buf368)
    buf369 = buf335; del buf335  # reuse
    buf370 = reinterpret_tensor(buf368, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf368  # reuse
    buf371 = buf370; del buf370  # reuse
    buf372 = buf333; del buf333  # reuse
    buf373 = buf371; del buf371  # reuse
    cpp_fused__softmax_52(c_void_p(buf373.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf372.data_ptr()))
    # Source Nodes: [attn_weights_21, softmax_10], Original ATen: [aten._softmax, aten.native_dropout]
    buf374 = aten.native_dropout(buf373, 0.1, True)
    buf375 = buf374[0]
    buf376 = buf374[1]
    del buf374
    buf377 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf367, (6, 128, 64), (64, 384, 1), 0), out=buf377)
    buf378 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_53(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf378, reinterpret_tensor(primals_116, (384, 512), (1, 384), 0), out=buf379)
    # Source Nodes: [l__mod___decoder_block_1_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf380 = aten.native_dropout(reinterpret_tensor(buf379, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf381 = buf380[0]
    buf382 = buf380[1]
    del buf380
    buf383 = buf381; del buf381  # reuse
    buf384 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf384, (1, 128, 1), (128, 1, 1), 0); del buf384  # reuse
    buf386 = buf379; del buf379  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_54(c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = reinterpret_tensor(buf377, (128, 384), (384, 1), 0); del buf377  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf386, reinterpret_tensor(primals_117, (512, 384), (1, 512), 0), out=buf387)
    buf388 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_118, (512, 384), (1, 512), 0), out=buf388)
    buf389 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 384), (1, 512), 0), out=buf389)
    buf390 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf387, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf388, (6, 64, 128), (64, 1, 384), 0), out=buf390)
    buf391 = buf372; del buf372  # reuse
    buf392 = reinterpret_tensor(buf390, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf390  # reuse
    buf393 = buf369; del buf369  # reuse
    buf394 = buf392; del buf392  # reuse
    cpp_fused__softmax_55(c_void_p(buf394.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf393.data_ptr()))
    # Source Nodes: [attn_weights_23, softmax_11], Original ATen: [aten._softmax, aten.native_dropout]
    buf395 = aten.native_dropout(buf394, 0.1, True)
    buf396 = buf395[0]
    buf397 = buf395[1]
    del buf395
    buf398 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf396, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf389, (6, 128, 64), (64, 384, 1), 0), out=buf398)
    buf399 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_56(c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(buf399, reinterpret_tensor(primals_120, (384, 512), (1, 384), 0), out=buf400)
    # Source Nodes: [l__mod___decoder_block_1_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf401 = aten.native_dropout(reinterpret_tensor(buf400, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf402 = buf401[0]
    buf403 = buf401[1]
    del buf401
    buf404 = buf402; del buf402  # reuse
    buf405 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf406 = reinterpret_tensor(buf405, (1, 128, 1), (128, 1, 1), 0); del buf405  # reuse
    buf407 = buf400; del buf400  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_57(c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf353, (128, 1024), (1024, 1), 0); del buf353  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf407, reinterpret_tensor(primals_121, (512, 1024), (1, 512), 0), out=buf408)
    buf410 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf407, reinterpret_tensor(primals_122, (512, 1024), (1, 512), 0), out=buf410)
    buf409 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf411 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_58(c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()))
    # Source Nodes: [add_70, hidden_gelu_9, hidden_states_128, hidden_states_129, mul_98], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf412 = aten.native_dropout(buf411, 0.1, True)
    buf413 = buf412[0]
    buf414 = buf412[1]
    del buf412
    buf415 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_123, (1024, 512), (1, 1024), 0), out=buf415)
    # Source Nodes: [l__mod___decoder_block_1_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf416 = aten.native_dropout(reinterpret_tensor(buf415, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf417 = buf416[0]
    buf418 = buf416[1]
    del buf416
    buf419 = buf417; del buf417  # reuse
    buf420 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf421 = reinterpret_tensor(buf420, (1, 128, 1), (128, 1, 1), 0); del buf420  # reuse
    buf422 = buf415; del buf415  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_59(c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf398, (128, 384), (384, 1), 0); del buf398  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, reinterpret_tensor(primals_124, (512, 384), (1, 512), 0), out=buf423)
    buf424 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, reinterpret_tensor(primals_125, (512, 384), (1, 512), 0), out=buf424)
    buf425 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, reinterpret_tensor(primals_126, (512, 384), (1, 512), 0), out=buf425)
    buf426 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf423, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf424, (6, 64, 128), (64, 1, 384), 0), out=buf426)
    buf427 = buf393; del buf393  # reuse
    buf428 = reinterpret_tensor(buf426, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf426  # reuse
    buf429 = buf428; del buf428  # reuse
    buf430 = buf391; del buf391  # reuse
    buf431 = buf429; del buf429  # reuse
    cpp_fused__softmax_60(c_void_p(buf431.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf430.data_ptr()))
    # Source Nodes: [attn_weights_25, softmax_12], Original ATen: [aten._softmax, aten.native_dropout]
    buf432 = aten.native_dropout(buf431, 0.1, True)
    buf433 = buf432[0]
    buf434 = buf432[1]
    del buf432
    buf435 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf433, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf425, (6, 128, 64), (64, 384, 1), 0), out=buf435)
    buf436 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_61(c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(buf436, reinterpret_tensor(primals_127, (384, 512), (1, 384), 0), out=buf437)
    # Source Nodes: [l__mod___decoder_block_2_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf438 = aten.native_dropout(reinterpret_tensor(buf437, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf439 = buf438[0]
    buf440 = buf438[1]
    del buf438
    buf441 = buf439; del buf439  # reuse
    buf442 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf443 = reinterpret_tensor(buf442, (1, 128, 1), (128, 1, 1), 0); del buf442  # reuse
    buf444 = buf437; del buf437  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_62(c_void_p(buf441.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = reinterpret_tensor(buf435, (128, 384), (384, 1), 0); del buf435  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf444, reinterpret_tensor(primals_128, (512, 384), (1, 512), 0), out=buf445)
    buf446 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 384), (1, 512), 0), out=buf446)
    buf447 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_130, (512, 384), (1, 512), 0), out=buf447)
    buf448 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf446, (6, 64, 128), (64, 1, 384), 0), out=buf448)
    buf449 = buf430; del buf430  # reuse
    buf450 = reinterpret_tensor(buf448, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf448  # reuse
    buf451 = buf427; del buf427  # reuse
    buf452 = buf450; del buf450  # reuse
    cpp_fused__softmax_63(c_void_p(buf452.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf451.data_ptr()))
    # Source Nodes: [attn_weights_27, softmax_13], Original ATen: [aten._softmax, aten.native_dropout]
    buf453 = aten.native_dropout(buf452, 0.1, True)
    buf454 = buf453[0]
    buf455 = buf453[1]
    del buf453
    buf456 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf454, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf447, (6, 128, 64), (64, 384, 1), 0), out=buf456)
    buf457 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_64(c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()))
    buf458 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf457, reinterpret_tensor(primals_131, (384, 512), (1, 384), 0), out=buf458)
    # Source Nodes: [l__mod___decoder_block_2_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf459 = aten.native_dropout(reinterpret_tensor(buf458, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf460 = buf459[0]
    buf461 = buf459[1]
    del buf459
    buf462 = buf460; del buf460  # reuse
    buf463 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf464 = reinterpret_tensor(buf463, (1, 128, 1), (128, 1, 1), 0); del buf463  # reuse
    buf465 = buf458; del buf458  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_65(c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf465.data_ptr()))
    buf466 = reinterpret_tensor(buf411, (128, 1024), (1024, 1), 0); del buf411  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf465, reinterpret_tensor(primals_132, (512, 1024), (1, 512), 0), out=buf466)
    buf468 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf465, reinterpret_tensor(primals_133, (512, 1024), (1, 512), 0), out=buf468)
    buf467 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf469 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_66(c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf469.data_ptr()))
    # Source Nodes: [add_78, hidden_gelu_10, hidden_states_144, hidden_states_145, mul_109], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf470 = aten.native_dropout(buf469, 0.1, True)
    buf471 = buf470[0]
    buf472 = buf470[1]
    del buf470
    buf473 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf471, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_134, (1024, 512), (1, 1024), 0), out=buf473)
    # Source Nodes: [l__mod___decoder_block_2_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf474 = aten.native_dropout(reinterpret_tensor(buf473, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf475 = buf474[0]
    buf476 = buf474[1]
    del buf474
    buf477 = buf475; del buf475  # reuse
    buf478 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf479 = reinterpret_tensor(buf478, (1, 128, 1), (128, 1, 1), 0); del buf478  # reuse
    buf480 = buf473; del buf473  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_67(c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf480.data_ptr()))
    buf481 = reinterpret_tensor(buf456, (128, 384), (384, 1), 0); del buf456  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf480, reinterpret_tensor(primals_135, (512, 384), (1, 512), 0), out=buf481)
    buf482 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf480, reinterpret_tensor(primals_136, (512, 384), (1, 512), 0), out=buf482)
    buf483 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf480, reinterpret_tensor(primals_137, (512, 384), (1, 512), 0), out=buf483)
    buf484 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf481, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf482, (6, 64, 128), (64, 1, 384), 0), out=buf484)
    buf485 = buf451; del buf451  # reuse
    buf486 = reinterpret_tensor(buf484, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf484  # reuse
    buf487 = buf486; del buf486  # reuse
    buf488 = buf449; del buf449  # reuse
    buf489 = buf487; del buf487  # reuse
    cpp_fused__softmax_68(c_void_p(buf489.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf488.data_ptr()))
    # Source Nodes: [attn_weights_29, softmax_14], Original ATen: [aten._softmax, aten.native_dropout]
    buf490 = aten.native_dropout(buf489, 0.1, True)
    buf491 = buf490[0]
    buf492 = buf490[1]
    del buf490
    buf493 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf491, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf483, (6, 128, 64), (64, 384, 1), 0), out=buf493)
    buf494 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_69(c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    buf495 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf494, reinterpret_tensor(primals_138, (384, 512), (1, 384), 0), out=buf495)
    # Source Nodes: [l__mod___decoder_block_3_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf496 = aten.native_dropout(reinterpret_tensor(buf495, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf497 = buf496[0]
    buf498 = buf496[1]
    del buf496
    buf499 = buf497; del buf497  # reuse
    buf500 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf501 = reinterpret_tensor(buf500, (1, 128, 1), (128, 1, 1), 0); del buf500  # reuse
    buf502 = buf495; del buf495  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_70(c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf502.data_ptr()))
    buf503 = reinterpret_tensor(buf493, (128, 384), (384, 1), 0); del buf493  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf502, reinterpret_tensor(primals_139, (512, 384), (1, 512), 0), out=buf503)
    buf504 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_140, (512, 384), (1, 512), 0), out=buf504)
    buf505 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_141, (512, 384), (1, 512), 0), out=buf505)
    buf506 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf503, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf504, (6, 64, 128), (64, 1, 384), 0), out=buf506)
    buf507 = buf488; del buf488  # reuse
    buf508 = reinterpret_tensor(buf506, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf506  # reuse
    buf509 = buf485; del buf485  # reuse
    buf510 = buf508; del buf508  # reuse
    cpp_fused__softmax_71(c_void_p(buf510.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()))
    # Source Nodes: [attn_weights_31, softmax_15], Original ATen: [aten._softmax, aten.native_dropout]
    buf511 = aten.native_dropout(buf510, 0.1, True)
    buf512 = buf511[0]
    buf513 = buf511[1]
    del buf511
    buf514 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf512, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf505, (6, 128, 64), (64, 384, 1), 0), out=buf514)
    buf515 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_72(c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()))
    buf516 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(buf515, reinterpret_tensor(primals_142, (384, 512), (1, 384), 0), out=buf516)
    # Source Nodes: [l__mod___decoder_block_3_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf517 = aten.native_dropout(reinterpret_tensor(buf516, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf518 = buf517[0]
    buf519 = buf517[1]
    del buf517
    buf520 = buf518; del buf518  # reuse
    buf521 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf522 = reinterpret_tensor(buf521, (1, 128, 1), (128, 1, 1), 0); del buf521  # reuse
    buf523 = buf516; del buf516  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_73(c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf523.data_ptr()))
    buf524 = reinterpret_tensor(buf469, (128, 1024), (1024, 1), 0); del buf469  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf523, reinterpret_tensor(primals_143, (512, 1024), (1, 512), 0), out=buf524)
    buf526 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_11], Original ATen: [aten.mm]
    extern_kernels.mm(buf523, reinterpret_tensor(primals_144, (512, 1024), (1, 512), 0), out=buf526)
    buf525 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf527 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_74(c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf527.data_ptr()))
    # Source Nodes: [add_86, hidden_gelu_11, hidden_states_160, hidden_states_161, mul_120], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf528 = aten.native_dropout(buf527, 0.1, True)
    buf529 = buf528[0]
    buf530 = buf528[1]
    del buf528
    buf531 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_145, (1024, 512), (1, 1024), 0), out=buf531)
    # Source Nodes: [l__mod___decoder_block_3_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf532 = aten.native_dropout(reinterpret_tensor(buf531, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf533 = buf532[0]
    buf534 = buf532[1]
    del buf532
    buf535 = buf533; del buf533  # reuse
    buf536 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf537 = reinterpret_tensor(buf536, (1, 128, 1), (128, 1, 1), 0); del buf536  # reuse
    buf538 = buf531; del buf531  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_75(c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = reinterpret_tensor(buf514, (128, 384), (384, 1), 0); del buf514  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf538, reinterpret_tensor(primals_146, (512, 384), (1, 512), 0), out=buf539)
    buf540 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf538, reinterpret_tensor(primals_147, (512, 384), (1, 512), 0), out=buf540)
    buf541 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf538, reinterpret_tensor(primals_148, (512, 384), (1, 512), 0), out=buf541)
    buf542 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf539, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf540, (6, 64, 128), (64, 1, 384), 0), out=buf542)
    buf543 = buf509; del buf509  # reuse
    buf544 = reinterpret_tensor(buf542, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf542  # reuse
    buf545 = buf544; del buf544  # reuse
    buf546 = buf507; del buf507  # reuse
    buf547 = buf545; del buf545  # reuse
    cpp_fused__softmax_76(c_void_p(buf547.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf546.data_ptr()))
    # Source Nodes: [attn_weights_33, softmax_16], Original ATen: [aten._softmax, aten.native_dropout]
    buf548 = aten.native_dropout(buf547, 0.1, True)
    buf549 = buf548[0]
    buf550 = buf548[1]
    del buf548
    buf551 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf549, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf541, (6, 128, 64), (64, 384, 1), 0), out=buf551)
    buf552 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_77(c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()))
    buf553 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf552, reinterpret_tensor(primals_149, (384, 512), (1, 384), 0), out=buf553)
    # Source Nodes: [l__mod___decoder_block_4_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf554 = aten.native_dropout(reinterpret_tensor(buf553, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf555 = buf554[0]
    buf556 = buf554[1]
    del buf554
    buf557 = buf555; del buf555  # reuse
    buf558 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf559 = reinterpret_tensor(buf558, (1, 128, 1), (128, 1, 1), 0); del buf558  # reuse
    buf560 = buf553; del buf553  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_78(c_void_p(buf557.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf560.data_ptr()))
    buf561 = reinterpret_tensor(buf551, (128, 384), (384, 1), 0); del buf551  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf560, reinterpret_tensor(primals_150, (512, 384), (1, 512), 0), out=buf561)
    buf562 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_151, (512, 384), (1, 512), 0), out=buf562)
    buf563 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_152, (512, 384), (1, 512), 0), out=buf563)
    buf564 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf561, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf562, (6, 64, 128), (64, 1, 384), 0), out=buf564)
    buf565 = buf546; del buf546  # reuse
    buf566 = reinterpret_tensor(buf564, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf564  # reuse
    buf567 = buf543; del buf543  # reuse
    buf568 = buf566; del buf566  # reuse
    cpp_fused__softmax_79(c_void_p(buf568.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf567.data_ptr()))
    # Source Nodes: [attn_weights_35, softmax_17], Original ATen: [aten._softmax, aten.native_dropout]
    buf569 = aten.native_dropout(buf568, 0.1, True)
    buf570 = buf569[0]
    buf571 = buf569[1]
    del buf569
    buf572 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf570, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf563, (6, 128, 64), (64, 384, 1), 0), out=buf572)
    buf573 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_80(c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(buf573, reinterpret_tensor(primals_153, (384, 512), (1, 384), 0), out=buf574)
    # Source Nodes: [l__mod___decoder_block_4_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf575 = aten.native_dropout(reinterpret_tensor(buf574, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf576 = buf575[0]
    buf577 = buf575[1]
    del buf575
    buf578 = buf576; del buf576  # reuse
    buf579 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf580 = reinterpret_tensor(buf579, (1, 128, 1), (128, 1, 1), 0); del buf579  # reuse
    buf581 = buf574; del buf574  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_81(c_void_p(buf578.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf581.data_ptr()))
    buf582 = reinterpret_tensor(buf527, (128, 1024), (1024, 1), 0); del buf527  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf581, reinterpret_tensor(primals_154, (512, 1024), (1, 512), 0), out=buf582)
    buf584 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf581, reinterpret_tensor(primals_155, (512, 1024), (1, 512), 0), out=buf584)
    buf583 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf585 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_82(c_void_p(buf582.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()))
    # Source Nodes: [add_94, hidden_gelu_12, hidden_states_176, hidden_states_177, mul_131], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf586 = aten.native_dropout(buf585, 0.1, True)
    buf587 = buf586[0]
    buf588 = buf586[1]
    del buf586
    buf589 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf587, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_156, (1024, 512), (1, 1024), 0), out=buf589)
    # Source Nodes: [l__mod___decoder_block_4_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf590 = aten.native_dropout(reinterpret_tensor(buf589, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf591 = buf590[0]
    buf592 = buf590[1]
    del buf590
    buf593 = buf591; del buf591  # reuse
    buf594 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf595 = reinterpret_tensor(buf594, (1, 128, 1), (128, 1, 1), 0); del buf594  # reuse
    buf596 = buf589; del buf589  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_83(c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf596.data_ptr()))
    buf597 = reinterpret_tensor(buf572, (128, 384), (384, 1), 0); del buf572  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf596, reinterpret_tensor(primals_157, (512, 384), (1, 512), 0), out=buf597)
    buf598 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf596, reinterpret_tensor(primals_158, (512, 384), (1, 512), 0), out=buf598)
    buf599 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf596, reinterpret_tensor(primals_159, (512, 384), (1, 512), 0), out=buf599)
    buf600 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf597, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf598, (6, 64, 128), (64, 1, 384), 0), out=buf600)
    buf601 = buf567; del buf567  # reuse
    buf602 = reinterpret_tensor(buf600, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf600  # reuse
    buf603 = buf602; del buf602  # reuse
    buf604 = buf565; del buf565  # reuse
    buf605 = buf603; del buf603  # reuse
    cpp_fused__softmax_84(c_void_p(buf605.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf604.data_ptr()))
    # Source Nodes: [attn_weights_37, softmax_18], Original ATen: [aten._softmax, aten.native_dropout]
    buf606 = aten.native_dropout(buf605, 0.1, True)
    buf607 = buf606[0]
    buf608 = buf606[1]
    del buf606
    buf609 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf607, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf599, (6, 128, 64), (64, 384, 1), 0), out=buf609)
    buf610 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_85(c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()))
    buf611 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_37], Original ATen: [aten.mm]
    extern_kernels.mm(buf610, reinterpret_tensor(primals_160, (384, 512), (1, 384), 0), out=buf611)
    # Source Nodes: [l__mod___decoder_block_5_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf612 = aten.native_dropout(reinterpret_tensor(buf611, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf613 = buf612[0]
    buf614 = buf612[1]
    del buf612
    buf615 = buf613; del buf613  # reuse
    buf616 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf617 = reinterpret_tensor(buf616, (1, 128, 1), (128, 1, 1), 0); del buf616  # reuse
    buf618 = buf611; del buf611  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_86(c_void_p(buf615.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf618.data_ptr()))
    buf619 = reinterpret_tensor(buf609, (128, 384), (384, 1), 0); del buf609  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf618, reinterpret_tensor(primals_161, (512, 384), (1, 512), 0), out=buf619)
    buf620 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_162, (512, 384), (1, 512), 0), out=buf620)
    buf621 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_163, (512, 384), (1, 512), 0), out=buf621)
    buf622 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf619, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf620, (6, 64, 128), (64, 1, 384), 0), out=buf622)
    buf623 = buf604; del buf604  # reuse
    buf624 = reinterpret_tensor(buf622, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf622  # reuse
    buf625 = buf601; del buf601  # reuse
    buf626 = buf624; del buf624  # reuse
    cpp_fused__softmax_87(c_void_p(buf626.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf625.data_ptr()))
    # Source Nodes: [attn_weights_39, softmax_19], Original ATen: [aten._softmax, aten.native_dropout]
    buf627 = aten.native_dropout(buf626, 0.1, True)
    buf628 = buf627[0]
    buf629 = buf627[1]
    del buf627
    buf630 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf628, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf621, (6, 128, 64), (64, 384, 1), 0), out=buf630)
    buf631 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_88(c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()))
    buf632 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_39], Original ATen: [aten.mm]
    extern_kernels.mm(buf631, reinterpret_tensor(primals_164, (384, 512), (1, 384), 0), out=buf632)
    # Source Nodes: [l__mod___decoder_block_5_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf633 = aten.native_dropout(reinterpret_tensor(buf632, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf634 = buf633[0]
    buf635 = buf633[1]
    del buf633
    buf636 = buf634; del buf634  # reuse
    buf637 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf638 = reinterpret_tensor(buf637, (1, 128, 1), (128, 1, 1), 0); del buf637  # reuse
    buf639 = buf632; del buf632  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_89(c_void_p(buf636.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf639.data_ptr()))
    buf640 = reinterpret_tensor(buf585, (128, 1024), (1024, 1), 0); del buf585  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf639, reinterpret_tensor(primals_165, (512, 1024), (1, 512), 0), out=buf640)
    buf642 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_13], Original ATen: [aten.mm]
    extern_kernels.mm(buf639, reinterpret_tensor(primals_166, (512, 1024), (1, 512), 0), out=buf642)
    buf641 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf643 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_90(c_void_p(buf640.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf643.data_ptr()))
    # Source Nodes: [add_102, hidden_gelu_13, hidden_states_192, hidden_states_193, mul_142], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf644 = aten.native_dropout(buf643, 0.1, True)
    buf645 = buf644[0]
    buf646 = buf644[1]
    del buf644
    buf647 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf645, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_167, (1024, 512), (1, 1024), 0), out=buf647)
    # Source Nodes: [l__mod___decoder_block_5_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf648 = aten.native_dropout(reinterpret_tensor(buf647, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf649 = buf648[0]
    buf650 = buf648[1]
    del buf648
    buf651 = buf649; del buf649  # reuse
    buf652 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf653 = reinterpret_tensor(buf652, (1, 128, 1), (128, 1, 1), 0); del buf652  # reuse
    buf654 = buf647; del buf647  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_91(c_void_p(buf651.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf654.data_ptr()))
    buf655 = reinterpret_tensor(buf630, (128, 384), (384, 1), 0); del buf630  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf654, reinterpret_tensor(primals_168, (512, 384), (1, 512), 0), out=buf655)
    buf656 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf654, reinterpret_tensor(primals_169, (512, 384), (1, 512), 0), out=buf656)
    buf657 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf654, reinterpret_tensor(primals_170, (512, 384), (1, 512), 0), out=buf657)
    buf658 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf655, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf656, (6, 64, 128), (64, 1, 384), 0), out=buf658)
    buf659 = buf625; del buf625  # reuse
    buf660 = reinterpret_tensor(buf658, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf658  # reuse
    buf661 = buf660; del buf660  # reuse
    buf662 = buf623; del buf623  # reuse
    buf663 = buf661; del buf661  # reuse
    cpp_fused__softmax_92(c_void_p(buf663.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf662.data_ptr()))
    # Source Nodes: [attn_weights_41, softmax_20], Original ATen: [aten._softmax, aten.native_dropout]
    buf664 = aten.native_dropout(buf663, 0.1, True)
    buf665 = buf664[0]
    buf666 = buf664[1]
    del buf664
    buf667 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf665, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf657, (6, 128, 64), (64, 384, 1), 0), out=buf667)
    buf668 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_93(c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()))
    buf669 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_41], Original ATen: [aten.mm]
    extern_kernels.mm(buf668, reinterpret_tensor(primals_171, (384, 512), (1, 384), 0), out=buf669)
    # Source Nodes: [l__mod___decoder_block_6_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf670 = aten.native_dropout(reinterpret_tensor(buf669, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf671 = buf670[0]
    buf672 = buf670[1]
    del buf670
    buf673 = buf671; del buf671  # reuse
    buf674 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf675 = reinterpret_tensor(buf674, (1, 128, 1), (128, 1, 1), 0); del buf674  # reuse
    buf676 = buf669; del buf669  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_94(c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf676.data_ptr()))
    buf677 = reinterpret_tensor(buf667, (128, 384), (384, 1), 0); del buf667  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf676, reinterpret_tensor(primals_172, (512, 384), (1, 512), 0), out=buf677)
    buf678 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_173, (512, 384), (1, 512), 0), out=buf678)
    buf679 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_174, (512, 384), (1, 512), 0), out=buf679)
    buf680 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf677, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf678, (6, 64, 128), (64, 1, 384), 0), out=buf680)
    buf681 = buf662; del buf662  # reuse
    buf682 = reinterpret_tensor(buf680, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf680  # reuse
    buf683 = buf659; del buf659  # reuse
    buf684 = buf682; del buf682  # reuse
    cpp_fused__softmax_95(c_void_p(buf684.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf683.data_ptr()))
    # Source Nodes: [attn_weights_43, softmax_21], Original ATen: [aten._softmax, aten.native_dropout]
    buf685 = aten.native_dropout(buf684, 0.1, True)
    buf686 = buf685[0]
    buf687 = buf685[1]
    del buf685
    buf688 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf686, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf679, (6, 128, 64), (64, 384, 1), 0), out=buf688)
    buf689 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_96(c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()))
    buf690 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_43], Original ATen: [aten.mm]
    extern_kernels.mm(buf689, reinterpret_tensor(primals_175, (384, 512), (1, 384), 0), out=buf690)
    # Source Nodes: [l__mod___decoder_block_6_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf691 = aten.native_dropout(reinterpret_tensor(buf690, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf692 = buf691[0]
    buf693 = buf691[1]
    del buf691
    buf694 = buf692; del buf692  # reuse
    buf695 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf696 = reinterpret_tensor(buf695, (1, 128, 1), (128, 1, 1), 0); del buf695  # reuse
    buf697 = buf690; del buf690  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_97(c_void_p(buf694.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf697.data_ptr()))
    buf698 = reinterpret_tensor(buf643, (128, 1024), (1024, 1), 0); del buf643  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf697, reinterpret_tensor(primals_176, (512, 1024), (1, 512), 0), out=buf698)
    buf700 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_14], Original ATen: [aten.mm]
    extern_kernels.mm(buf697, reinterpret_tensor(primals_177, (512, 1024), (1, 512), 0), out=buf700)
    buf699 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf701 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_98(c_void_p(buf698.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf701.data_ptr()))
    # Source Nodes: [add_110, hidden_gelu_14, hidden_states_208, hidden_states_209, mul_153], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf702 = aten.native_dropout(buf701, 0.1, True)
    buf703 = buf702[0]
    buf704 = buf702[1]
    del buf702
    buf705 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf703, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_178, (1024, 512), (1, 1024), 0), out=buf705)
    # Source Nodes: [l__mod___decoder_block_6_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf706 = aten.native_dropout(reinterpret_tensor(buf705, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf707 = buf706[0]
    buf708 = buf706[1]
    del buf706
    buf709 = buf707; del buf707  # reuse
    buf710 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf711 = reinterpret_tensor(buf710, (1, 128, 1), (128, 1, 1), 0); del buf710  # reuse
    buf712 = buf705; del buf705  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_99(c_void_p(buf709.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf712.data_ptr()))
    buf713 = reinterpret_tensor(buf688, (128, 384), (384, 1), 0); del buf688  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf712, reinterpret_tensor(primals_179, (512, 384), (1, 512), 0), out=buf713)
    buf714 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(buf712, reinterpret_tensor(primals_180, (512, 384), (1, 512), 0), out=buf714)
    buf715 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf712, reinterpret_tensor(primals_181, (512, 384), (1, 512), 0), out=buf715)
    buf716 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf713, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf714, (6, 64, 128), (64, 1, 384), 0), out=buf716)
    buf717 = buf683; del buf683  # reuse
    buf718 = reinterpret_tensor(buf716, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf716  # reuse
    buf719 = buf718; del buf718  # reuse
    buf720 = buf681; del buf681  # reuse
    buf721 = buf719; del buf719  # reuse
    cpp_fused__softmax_100(c_void_p(buf721.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf720.data_ptr()))
    del primals_104
    # Source Nodes: [attn_weights_45, softmax_22], Original ATen: [aten._softmax, aten.native_dropout]
    buf722 = aten.native_dropout(buf721, 0.1, True)
    buf723 = buf722[0]
    buf724 = buf722[1]
    del buf722
    buf725 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf723, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf715, (6, 128, 64), (64, 384, 1), 0), out=buf725)
    buf726 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_101(c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()))
    buf727 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf726, reinterpret_tensor(primals_182, (384, 512), (1, 384), 0), out=buf727)
    # Source Nodes: [l__mod___decoder_block_7_layer_0_dropout], Original ATen: [aten.native_dropout]
    buf728 = aten.native_dropout(reinterpret_tensor(buf727, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf729 = buf728[0]
    buf730 = buf728[1]
    del buf728
    buf731 = buf729; del buf729  # reuse
    buf732 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf733 = reinterpret_tensor(buf732, (1, 128, 1), (128, 1, 1), 0); del buf732  # reuse
    buf734 = buf727; del buf727  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_102(c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf734.data_ptr()))
    buf735 = reinterpret_tensor(buf725, (128, 384), (384, 1), 0); del buf725  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(buf734, reinterpret_tensor(primals_183, (512, 384), (1, 512), 0), out=buf735)
    buf736 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_184, (512, 384), (1, 512), 0), out=buf736)
    buf737 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (128, 512), (512, 1), 0), reinterpret_tensor(primals_185, (512, 384), (1, 512), 0), out=buf737)
    buf738 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf735, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf736, (6, 64, 128), (64, 1, 384), 0), out=buf738)
    buf739 = buf720; del buf720  # reuse
    buf740 = reinterpret_tensor(buf738, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf738  # reuse
    buf741 = buf717; del buf717  # reuse
    buf742 = buf740; del buf740  # reuse
    cpp_fused__softmax_103(c_void_p(buf742.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf741.data_ptr()))
    del buf739
    del buf741
    # Source Nodes: [attn_weights_47, softmax_23], Original ATen: [aten._softmax, aten.native_dropout]
    buf743 = aten.native_dropout(buf742, 0.1, True)
    buf744 = buf743[0]
    buf745 = buf743[1]
    del buf743
    buf746 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf744, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf737, (6, 128, 64), (64, 384, 1), 0), out=buf746)
    buf747 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_104(c_void_p(buf746.data_ptr()), c_void_p(buf747.data_ptr()))
    del buf746
    buf748 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_47], Original ATen: [aten.mm]
    extern_kernels.mm(buf747, reinterpret_tensor(primals_186, (384, 512), (1, 384), 0), out=buf748)
    # Source Nodes: [l__mod___decoder_block_7_layer_1_dropout], Original ATen: [aten.native_dropout]
    buf749 = aten.native_dropout(reinterpret_tensor(buf748, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf750 = buf749[0]
    buf751 = buf749[1]
    del buf749
    buf752 = buf750; del buf750  # reuse
    buf753 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf754 = reinterpret_tensor(buf753, (1, 128, 1), (128, 1, 1), 0); del buf753  # reuse
    buf755 = buf748; del buf748  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_view_105(c_void_p(buf752.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf755.data_ptr()))
    buf756 = reinterpret_tensor(buf701, (128, 1024), (1024, 1), 0); del buf701  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(buf755, reinterpret_tensor(primals_187, (512, 1024), (1, 512), 0), out=buf756)
    buf758 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf755, reinterpret_tensor(primals_188, (512, 1024), (1, 512), 0), out=buf758)
    buf757 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf759 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_106(c_void_p(buf756.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf759.data_ptr()))
    # Source Nodes: [add_118, hidden_gelu_15, hidden_states_224, hidden_states_225, mul_164], Original ATen: [aten.add, aten.mul, aten.native_dropout]
    buf760 = aten.native_dropout(buf759, 0.1, True)
    del buf759
    buf761 = buf760[0]
    buf762 = buf760[1]
    del buf760
    buf763 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [forwarded_states_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf761, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_189, (1024, 512), (1, 1024), 0), out=buf763)
    # Source Nodes: [l__mod___decoder_block_7_layer__1__dropout], Original ATen: [aten.native_dropout]
    buf764 = aten.native_dropout(reinterpret_tensor(buf763, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf765 = buf764[0]
    buf766 = buf764[1]
    del buf764
    buf767 = buf765; del buf765  # reuse
    buf768 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf769 = reinterpret_tensor(buf768, (1, 128, 1), (128, 1, 1), 0); del buf768  # reuse
    buf770 = reinterpret_tensor(buf763, (1, 128, 512), (65536, 512, 1), 0); del buf763  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_107(c_void_p(buf767.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf770.data_ptr()))
    # Source Nodes: [hidden_states_230, hidden_states_231, sequence_output], Original ATen: [aten.mul, aten.native_dropout]
    buf771 = aten.native_dropout(buf770, 0.1, True)
    del buf770
    buf772 = buf771[0]
    buf773 = buf771[1]
    del buf771
    buf774 = empty((128, 250112), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf772, (128, 512), (512, 1), 0), reinterpret_tensor(primals_190, (512, 250112), (1, 512), 0), out=buf774)
    buf775 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf776 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf777 = empty((128, 250112), device='cpu', dtype=torch.float32)
    buf778 = empty((), device='cpu', dtype=torch.int64)
    buf780 = empty((), device='cpu', dtype=torch.float32)
    buf779 = empty((), device='cpu', dtype=torch.float32)
    buf781 = buf780; del buf780  # reuse
    cpp_fused__log_softmax_nll_loss_forward_108(c_void_p(buf781.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf779.data_ptr()))
    return (buf781, reinterpret_tensor(buf774, (1, 128, 250112), (32014336, 250112, 1), 0), reinterpret_tensor(buf307, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf308, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf330, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf331, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf366, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf367, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf388, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf389, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf424, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf425, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf446, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf447, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf482, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf483, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf504, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf505, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf540, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf541, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf562, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf563, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf598, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf599, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf620, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf621, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf656, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf657, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf678, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf679, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf714, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf715, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf736, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf737, (1, 6, 128, 64), (49152, 64, 384, 1), 0), buf297, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, primals_191, buf2, buf3, buf5, buf6, buf11, buf18, buf20, buf24, buf25, buf27, buf28, buf29, buf30, buf31, buf35, reinterpret_tensor(buf34, (128, 1024), (1024, 1), 0), buf39, buf40, buf42, buf43, buf54, buf56, buf60, buf61, buf63, buf64, buf65, buf66, buf67, buf71, reinterpret_tensor(buf70, (128, 1024), (1024, 1), 0), buf75, buf76, buf78, buf79, buf90, buf92, buf96, buf97, buf99, buf100, buf101, buf102, buf103, buf107, reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), buf111, buf112, buf114, buf115, buf126, buf128, buf132, buf133, buf135, buf136, buf137, buf138, buf139, buf143, reinterpret_tensor(buf142, (128, 1024), (1024, 1), 0), buf147, buf148, buf150, buf151, buf162, buf164, buf168, buf169, buf171, buf172, buf173, buf174, buf175, buf179, reinterpret_tensor(buf178, (128, 1024), (1024, 1), 0), buf183, buf184, buf186, buf187, buf198, buf200, buf204, buf205, buf207, buf208, buf209, buf210, buf211, buf215, reinterpret_tensor(buf214, (128, 1024), (1024, 1), 0), buf219, buf220, buf222, buf223, buf234, buf236, buf240, buf241, buf243, buf244, buf245, buf246, buf247, buf251, reinterpret_tensor(buf250, (128, 1024), (1024, 1), 0), buf255, buf256, buf258, buf259, buf270, buf272, buf276, buf277, buf279, buf280, buf281, buf282, buf283, buf287, reinterpret_tensor(buf286, (128, 1024), (1024, 1), 0), buf291, buf292, buf294, buf298, primals_193, buf301, buf302, buf304, buf305, buf310, buf318, buf320, buf324, buf325, buf327, buf328, reinterpret_tensor(buf297, (128, 512), (512, 1), 0), buf339, buf341, buf345, buf346, buf348, buf349, buf350, buf351, buf352, buf356, reinterpret_tensor(buf355, (128, 1024), (1024, 1), 0), buf360, buf361, buf363, buf364, buf376, buf378, buf382, buf383, buf385, buf386, buf397, buf399, buf403, buf404, buf406, buf407, buf408, buf409, buf410, buf414, reinterpret_tensor(buf413, (128, 1024), (1024, 1), 0), buf418, buf419, buf421, buf422, buf434, buf436, buf440, buf441, buf443, buf444, buf455, buf457, buf461, buf462, buf464, buf465, buf466, buf467, buf468, buf472, reinterpret_tensor(buf471, (128, 1024), (1024, 1), 0), buf476, buf477, buf479, buf480, buf492, buf494, buf498, buf499, buf501, buf502, buf513, buf515, buf519, buf520, buf522, buf523, buf524, buf525, buf526, buf530, reinterpret_tensor(buf529, (128, 1024), (1024, 1), 0), buf534, buf535, buf537, buf538, buf550, buf552, buf556, buf557, buf559, buf560, buf571, buf573, buf577, buf578, buf580, buf581, buf582, buf583, buf584, buf588, reinterpret_tensor(buf587, (128, 1024), (1024, 1), 0), buf592, buf593, buf595, buf596, buf608, buf610, buf614, buf615, buf617, buf618, buf629, buf631, buf635, buf636, buf638, buf639, buf640, buf641, buf642, buf646, reinterpret_tensor(buf645, (128, 1024), (1024, 1), 0), buf650, buf651, buf653, buf654, buf666, buf668, buf672, buf673, buf675, buf676, buf687, buf689, buf693, buf694, buf696, buf697, buf698, buf699, buf700, buf704, reinterpret_tensor(buf703, (128, 1024), (1024, 1), 0), buf708, buf709, buf711, buf712, buf724, buf726, buf730, buf731, buf733, buf734, buf745, buf747, buf751, buf752, buf754, buf755, buf756, buf757, buf758, buf762, reinterpret_tensor(buf761, (128, 1024), (1024, 1), 0), buf766, buf767, buf769, buf773, reinterpret_tensor(buf772, (128, 512), (512, 1), 0), buf777, buf779, reinterpret_tensor(primals_190, (250112, 512), (512, 1), 0), reinterpret_tensor(primals_189, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_188, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_187, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_186, (512, 384), (384, 1), 0), reinterpret_tensor(buf744, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf737, (6, 64, 128), (64, 1, 384), 0), buf742, reinterpret_tensor(buf735, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf736, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_185, (384, 512), (512, 1), 0), reinterpret_tensor(primals_184, (384, 512), (512, 1), 0), reinterpret_tensor(primals_183, (384, 512), (512, 1), 0), reinterpret_tensor(primals_182, (512, 384), (384, 1), 0), reinterpret_tensor(buf723, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf715, (6, 64, 128), (64, 1, 384), 0), buf721, reinterpret_tensor(buf713, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf714, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_181, (384, 512), (512, 1), 0), reinterpret_tensor(primals_180, (384, 512), (512, 1), 0), reinterpret_tensor(primals_179, (384, 512), (512, 1), 0), reinterpret_tensor(primals_178, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_177, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_176, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_175, (512, 384), (384, 1), 0), reinterpret_tensor(buf686, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf679, (6, 64, 128), (64, 1, 384), 0), buf684, reinterpret_tensor(buf677, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf678, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_174, (384, 512), (512, 1), 0), reinterpret_tensor(primals_173, (384, 512), (512, 1), 0), reinterpret_tensor(primals_172, (384, 512), (512, 1), 0), reinterpret_tensor(primals_171, (512, 384), (384, 1), 0), reinterpret_tensor(buf665, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf657, (6, 64, 128), (64, 1, 384), 0), buf663, reinterpret_tensor(buf655, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf656, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_170, (384, 512), (512, 1), 0), reinterpret_tensor(primals_169, (384, 512), (512, 1), 0), reinterpret_tensor(primals_168, (384, 512), (512, 1), 0), reinterpret_tensor(primals_167, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_166, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_165, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_164, (512, 384), (384, 1), 0), reinterpret_tensor(buf628, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf621, (6, 64, 128), (64, 1, 384), 0), buf626, reinterpret_tensor(buf619, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf620, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_163, (384, 512), (512, 1), 0), reinterpret_tensor(primals_162, (384, 512), (512, 1), 0), reinterpret_tensor(primals_161, (384, 512), (512, 1), 0), reinterpret_tensor(primals_160, (512, 384), (384, 1), 0), reinterpret_tensor(buf607, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf599, (6, 64, 128), (64, 1, 384), 0), buf605, reinterpret_tensor(buf597, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf598, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_159, (384, 512), (512, 1), 0), reinterpret_tensor(primals_158, (384, 512), (512, 1), 0), reinterpret_tensor(primals_157, (384, 512), (512, 1), 0), reinterpret_tensor(primals_156, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_155, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_154, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_153, (512, 384), (384, 1), 0), reinterpret_tensor(buf570, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf563, (6, 64, 128), (64, 1, 384), 0), buf568, reinterpret_tensor(buf561, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf562, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_152, (384, 512), (512, 1), 0), reinterpret_tensor(primals_151, (384, 512), (512, 1), 0), reinterpret_tensor(primals_150, (384, 512), (512, 1), 0), reinterpret_tensor(primals_149, (512, 384), (384, 1), 0), reinterpret_tensor(buf549, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf541, (6, 64, 128), (64, 1, 384), 0), buf547, reinterpret_tensor(buf539, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf540, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_148, (384, 512), (512, 1), 0), reinterpret_tensor(primals_147, (384, 512), (512, 1), 0), reinterpret_tensor(primals_146, (384, 512), (512, 1), 0), reinterpret_tensor(primals_145, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_144, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_143, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_142, (512, 384), (384, 1), 0), reinterpret_tensor(buf512, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf505, (6, 64, 128), (64, 1, 384), 0), buf510, reinterpret_tensor(buf503, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf504, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_141, (384, 512), (512, 1), 0), reinterpret_tensor(primals_140, (384, 512), (512, 1), 0), reinterpret_tensor(primals_139, (384, 512), (512, 1), 0), reinterpret_tensor(primals_138, (512, 384), (384, 1), 0), reinterpret_tensor(buf491, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf483, (6, 64, 128), (64, 1, 384), 0), buf489, reinterpret_tensor(buf481, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf482, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_137, (384, 512), (512, 1), 0), reinterpret_tensor(primals_136, (384, 512), (512, 1), 0), reinterpret_tensor(primals_135, (384, 512), (512, 1), 0), reinterpret_tensor(primals_134, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_133, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_132, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_131, (512, 384), (384, 1), 0), reinterpret_tensor(buf454, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf447, (6, 64, 128), (64, 1, 384), 0), buf452, reinterpret_tensor(buf445, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf446, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_130, (384, 512), (512, 1), 0), reinterpret_tensor(primals_129, (384, 512), (512, 1), 0), reinterpret_tensor(primals_128, (384, 512), (512, 1), 0), reinterpret_tensor(primals_127, (512, 384), (384, 1), 0), reinterpret_tensor(buf433, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf425, (6, 64, 128), (64, 1, 384), 0), buf431, reinterpret_tensor(buf423, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf424, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_126, (384, 512), (512, 1), 0), reinterpret_tensor(primals_125, (384, 512), (512, 1), 0), reinterpret_tensor(primals_124, (384, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_122, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_121, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 384), (384, 1), 0), reinterpret_tensor(buf396, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf389, (6, 64, 128), (64, 1, 384), 0), buf394, reinterpret_tensor(buf387, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf388, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_119, (384, 512), (512, 1), 0), reinterpret_tensor(primals_118, (384, 512), (512, 1), 0), reinterpret_tensor(primals_117, (384, 512), (512, 1), 0), reinterpret_tensor(primals_116, (512, 384), (384, 1), 0), reinterpret_tensor(buf375, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf367, (6, 64, 128), (64, 1, 384), 0), buf373, reinterpret_tensor(buf365, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf366, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_115, (384, 512), (512, 1), 0), reinterpret_tensor(primals_114, (384, 512), (512, 1), 0), reinterpret_tensor(primals_113, (384, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_111, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_110, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 384), (384, 1), 0), reinterpret_tensor(buf338, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf331, (6, 64, 128), (64, 1, 384), 0), buf336, reinterpret_tensor(buf329, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf330, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_108, (384, 512), (512, 1), 0), reinterpret_tensor(primals_107, (384, 512), (512, 1), 0), reinterpret_tensor(primals_106, (384, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 384), (384, 1), 0), reinterpret_tensor(buf317, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf308, (6, 64, 128), (64, 1, 384), 0), buf315, reinterpret_tensor(buf306, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf307, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_103, (384, 512), (512, 1), 0), reinterpret_tensor(primals_102, (384, 512), (512, 1), 0), reinterpret_tensor(primals_101, (384, 512), (512, 1), 0), reinterpret_tensor(primals_100, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_99, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_98, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 384), (384, 1), 0), reinterpret_tensor(buf269, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf262, (6, 64, 128), (64, 1, 384), 0), buf267, reinterpret_tensor(buf260, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf261, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_96, (384, 512), (512, 1), 0), reinterpret_tensor(primals_95, (384, 512), (512, 1), 0), reinterpret_tensor(primals_94, (384, 512), (512, 1), 0), reinterpret_tensor(primals_93, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_92, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_91, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_90, (512, 384), (384, 1), 0), reinterpret_tensor(buf233, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf226, (6, 64, 128), (64, 1, 384), 0), buf231, reinterpret_tensor(buf224, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf225, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_89, (384, 512), (512, 1), 0), reinterpret_tensor(primals_88, (384, 512), (512, 1), 0), reinterpret_tensor(primals_87, (384, 512), (512, 1), 0), reinterpret_tensor(primals_86, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_85, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_84, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_83, (512, 384), (384, 1), 0), reinterpret_tensor(buf197, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf190, (6, 64, 128), (64, 1, 384), 0), buf195, reinterpret_tensor(buf188, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf189, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_82, (384, 512), (512, 1), 0), reinterpret_tensor(primals_81, (384, 512), (512, 1), 0), reinterpret_tensor(primals_80, (384, 512), (512, 1), 0), reinterpret_tensor(primals_79, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_78, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_77, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_76, (512, 384), (384, 1), 0), reinterpret_tensor(buf161, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf154, (6, 64, 128), (64, 1, 384), 0), buf159, reinterpret_tensor(buf152, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf153, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_75, (384, 512), (512, 1), 0), reinterpret_tensor(primals_74, (384, 512), (512, 1), 0), reinterpret_tensor(primals_73, (384, 512), (512, 1), 0), reinterpret_tensor(primals_72, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_71, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_70, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_69, (512, 384), (384, 1), 0), reinterpret_tensor(buf125, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf118, (6, 64, 128), (64, 1, 384), 0), buf123, reinterpret_tensor(buf116, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf117, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_68, (384, 512), (512, 1), 0), reinterpret_tensor(primals_67, (384, 512), (512, 1), 0), reinterpret_tensor(primals_66, (384, 512), (512, 1), 0), reinterpret_tensor(primals_65, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_64, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_63, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_62, (512, 384), (384, 1), 0), reinterpret_tensor(buf89, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf82, (6, 64, 128), (64, 1, 384), 0), buf87, reinterpret_tensor(buf80, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf81, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_61, (384, 512), (512, 1), 0), reinterpret_tensor(primals_60, (384, 512), (512, 1), 0), reinterpret_tensor(primals_59, (384, 512), (512, 1), 0), reinterpret_tensor(primals_58, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_57, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_56, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_55, (512, 384), (384, 1), 0), reinterpret_tensor(buf53, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf46, (6, 64, 128), (64, 1, 384), 0), buf51, reinterpret_tensor(buf44, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf45, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_54, (384, 512), (512, 1), 0), reinterpret_tensor(primals_53, (384, 512), (512, 1), 0), reinterpret_tensor(primals_52, (384, 512), (512, 1), 0), reinterpret_tensor(primals_51, (512, 1024), (1024, 1), 0), reinterpret_tensor(primals_50, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_49, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_48, (512, 384), (384, 1), 0), reinterpret_tensor(buf17, (6, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf9, (6, 64, 128), (64, 1, 384), 0), buf15, reinterpret_tensor(buf7, (6, 64, 128), (64, 1, 384), 0), reinterpret_tensor(buf8, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(primals_46, (384, 512), (512, 1), 0), reinterpret_tensor(primals_45, (384, 512), (512, 1), 0), reinterpret_tensor(primals_44, (384, 512), (512, 1), 0), )


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
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((250112, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((32, 6), (6, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((32, 6), (6, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((250112, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_193 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
